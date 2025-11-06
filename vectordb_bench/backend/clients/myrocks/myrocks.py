import csv
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import mysql.connector
import numpy as np

from ..api import VectorDB
from .config import MyRocksConfigDict, MyRocksIndexConfig

log = logging.getLogger(__name__)


class MyRocks(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: MyRocksConfigDict,
        db_case_config: MyRocksIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "MyRocks"
        self.db_config = db_config
        self.case_config = db_case_config
        self.db_name = "vectordbbench"
        self.table_name = collection_name
        self.dim = dim

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        if drop_old:
            self._drop_db()
            self._create_db_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[mysql.connector.MySQLConnection, mysql.connector.cursor.MySQLCursor]:
        conn = mysql.connector.connect(**kwargs)
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _drop_db(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop db : {self.db_name}")

        # flush tables before dropping database to avoid some locking issue
        self.cursor.execute("FLUSH TABLES")
        self.cursor.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
        self.conn.commit()
        self.cursor.execute("FLUSH TABLES")

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create database : {self.db_name}")
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")

            log.info(f"{self.name} client use database : {self.db_name}")
            self.cursor.execute(f"USE {self.db_name}")

            # Note: Centroids are loaded by RocksDB at C++ level from
            # /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256.csv
            # No need to create a centroid table in SQL

            # Create main vector table with LSM vector index
            # MyRocks requires the index to be created with the table, not via ALTER TABLE
            log.info(f"{self.name} client create table : {self.table_name}")
            self.cursor.execute(
                f"""
              CREATE TABLE {self.table_name} (
                id INT NOT NULL,
                v JSON NOT NULL FB_VECTOR_DIMENSION {dim},
                PRIMARY KEY (id) COMMENT 'cfname=cf1',
                INDEX v_idx(v) FB_VECTOR_INDEX_TYPE 'lsmidx' COMMENT 'cfname=cf1'
              ) ENGINE=ROCKSDB
            """
            )
            self.conn.commit()

        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise e from None

    @contextmanager
    def init(self):
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # maximize allowed package size for large vector batches
        self.cursor.execute("SET GLOBAL max_allowed_packet = 1073741824")
        self.conn.commit()

        # Set nprobe parameter for vector search if available in case_config
        nprobe = self.case_config.search_param().get("nprobe", 16)
        self.cursor.execute(f"SET SESSION fb_vector_search_nprobe = {nprobe}")
        self.conn.commit()
        log.info(f"Set fb_vector_search_nprobe = {nprobe}")

        # Prepare insert SQL statement
        # Using CAST to JSON as shown in the example SQL
        self.insert_sql = f"INSERT INTO {self.db_name}.{self.table_name} (id, v) VALUES (%s, CAST(%s AS JSON))"  # noqa: S608

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def ready_to_load(self) -> bool:
        pass

    def _generate_random_centroids(self, num_centroids: int = 256) -> None:
        """Generate and insert random centroids for testing purposes.

        In production, centroids should be pre-computed using k-means clustering
        and loaded from a CSV file. This method is for testing only.

        Args:
            num_centroids: Number of random centroids to generate (default: 256)
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        centroid_table_name = f"{self.table_name}_centroids"

        log.info(f"Generating {num_centroids} random centroids for table {centroid_table_name}")

        try:
            # Generate random centroids with values in range [0, 1]
            centroids = np.random.rand(num_centroids, self.dim).astype(np.float32)

            # Prepare batch insert
            batch_data = []
            for i, centroid in enumerate(centroids):
                centroid_json = self.vector_to_json(centroid)
                batch_data.append((i, centroid_json))

            # Insert centroids
            insert_sql = f"INSERT INTO {self.db_name}.{centroid_table_name} (id, centroid) VALUES (%s, CAST(%s AS JSON))"  # noqa: S608
            self.cursor.executemany(insert_sql, batch_data)
            self.conn.commit()

            log.info(f"Successfully inserted {num_centroids} random centroids")

        except Exception as e:
            log.warning(f"Failed to insert centroids: {e}")
            raise e from None

    def _load_centroids_from_csv(self, csv_path: str) -> None:
        """Load pre-computed centroids from a CSV file.

        The CSV file should have columns: id, centroid
        where centroid is a JSON array string representation of the vector.

        Args:
            csv_path: Path to the CSV file containing centroids
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        centroid_table_name = f"{self.table_name}_centroids"
        csv_file = Path(csv_path)

        if not csv_file.exists():
            msg = f"Centroid CSV file not found: {csv_path}"
            log.error(msg)
            raise FileNotFoundError(msg)

        log.info(f"Loading centroids from {csv_path} into table {centroid_table_name}")

        try:
            batch_data = []
            with csv_file.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    centroid_id = int(row["id"])
                    # Assume centroid is already in JSON format or convert it
                    centroid_value = row["centroid"]

                    # If it's already a JSON string, use it directly
                    # Otherwise, parse and re-encode
                    try:
                        # Verify it's valid JSON
                        json.loads(centroid_value)
                        batch_data.append((centroid_id, centroid_value))
                    except json.JSONDecodeError:
                        log.warning(f"Invalid JSON in centroid id {centroid_id}, skipping")
                        continue

            # Insert centroids in batches
            insert_sql = f"INSERT INTO {self.db_name}.{centroid_table_name} (id, centroid) VALUES (%s, CAST(%s AS JSON))"  # noqa: S608
            self.cursor.executemany(insert_sql, batch_data)
            self.conn.commit()

            log.info(f"Successfully loaded {len(batch_data)} centroids from CSV")

        except Exception as e:
            log.warning(f"Failed to load centroids from CSV: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # LSM index is already created with the table in _create_db_table()
        # Centroids are loaded from /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256.csv
        # via block_based_table_factory.h SetIndexOptions() method at RocksDB C++ level

        log.info(f"LSM vector index already exists for {self.table_name} (created with table)")
        log.info(f"Centroids are loaded by RocksDB from C++ code at: /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256.csv")

        # Flush tables to ensure data is written to RocksDB
        self.cursor.execute("FLUSH TABLES")
        self.conn.commit()
        log.info(f"Tables flushed - optimize complete")

    @staticmethod
    def vector_to_json(v):  # noqa: ANN001
        """Convert vector to JSON string for MyRocks storage"""
        # MyRocks expects vectors as JSON arrays
        return json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            batch_data = []
            for i, row in enumerate(metadata_arr):
                # Convert vector to JSON format as required by MyRocks
                vector_json = self.vector_to_json(embeddings_arr[i])
                batch_data.append((int(row), vector_json))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.conn.commit()
            self.cursor.execute("FLUSH TABLES")

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        # Convert query vector to JSON string format
        query_json = json.dumps(query)

        # Use MyRocks FB_VECTOR_L2 function for L2 distance search
        # Based on the search queries provided, the syntax is:
        # SELECT id, FB_VECTOR_L2(table.column, '[vector]') AS dis FROM table ORDER BY dis ASC LIMIT k
        # Note: MyRocks currently only supports L2 distance
        distance_func = "FB_VECTOR_L2"

        # Build the search query
        if filters:
            # Apply filters if provided
            filter_clause = f"WHERE id >= {filters.get('id', 0)}"
            search_sql = f"""
                SELECT id, {distance_func}({self.table_name}.v, '{query_json}') AS dis
                FROM {self.db_name}.{self.table_name}
                {filter_clause}
                ORDER BY dis ASC
                LIMIT {k}
            """
        else:
            search_sql = f"""
                SELECT id, {distance_func}({self.table_name}.v, '{query_json}') AS dis
                FROM {self.db_name}.{self.table_name}
                ORDER BY dis ASC
                LIMIT {k}
            """

        self.cursor.execute(search_sql)
        results = self.cursor.fetchall()

        # Return only the IDs
        return [int(row[0]) for row in results]
