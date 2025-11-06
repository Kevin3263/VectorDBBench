import logging
import os
from contextlib import contextmanager

import duckdb
import numpy as np

from ..api import VectorDB
from .config import DuckDBConfigDict, DuckDBIndexConfig

log = logging.getLogger(__name__)


class DuckDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: DuckDBConfigDict,
        db_case_config: DuckDBIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "DuckDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        # Get threads configuration from kwargs if provided
        self.threads = kwargs.get("threads", None)

        # Track whether we're in read-only mode for concurrent access
        # Auto-detect: if we're not dropping old data, enable read-only mode for concurrent searches
        # This prevents file locking conflicts when multiple processes search simultaneously
        self._force_readonly = not drop_old and self._database_exists()

        # Create initial connection to set up database
        self.conn = self._create_connection(**self.db_config)

        # Install and load VSS extension
        self._setup_vss_extension()

        if drop_old:
            self._drop_table()
            # If we dropped old data, we're in write mode (loading data)
            self._force_readonly = False

        self._create_table(dim)

        # Close initial connection
        self.conn.close()
        self.conn = None

    def _database_exists(self) -> bool:
        """Check if the database file exists (not applicable for :memory: databases)"""
        db_path = self.db_config.get("database", ":memory:")
        if db_path == ":memory:":
            return False
        return os.path.exists(db_path) and os.path.getsize(db_path) > 0

    @staticmethod
    def _create_connection(**kwargs) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection"""
        conn = duckdb.connect(
            database=kwargs.get("database", ":memory:"),
            read_only=kwargs.get("read_only", False),
        )

        assert conn is not None, "Connection is not initialized"
        return conn

    def _setup_vss_extension(self):
        """Install and load the VSS (Vector Similarity Search) extension"""
        assert self.conn is not None, "Connection is not initialized"

        try:
            # Install VSS extension if not already installed
            self.conn.execute("INSTALL vss")
            log.info("VSS extension installed")
        except Exception as e:
            # Extension might already be installed
            log.debug(f"VSS extension install message: {e}")

        # Load VSS extension
        self.conn.execute("LOAD vss")
        log.info("VSS extension loaded")

        # Enable experimental persistence for HNSW indexes in persistent databases
        # This is required for creating HNSW indexes in file-based databases
        self.conn.execute("SET hnsw_enable_experimental_persistence = true")
        log.info("Enabled HNSW experimental persistence")

        # Set threads if specified
        if self.threads is not None:
            self.conn.execute(f"SET threads = {self.threads}")
            log.info(f"Set DuckDB threads to {self.threads}")

    def _drop_table(self):
        """Drop the existing table if it exists"""
        assert self.conn is not None, "Connection is not initialized"
        log.info(f"{self.name} client drop table: {self.table_name}")

        self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    def _create_table(self, dim: int):
        """Create table with vector column

        DuckDB uses FLOAT arrays for vector storage with fixed dimensions
        """
        assert self.conn is not None, "Connection is not initialized"

        try:
            log.info(f"{self.name} client create table: {self.table_name}")

            # Create table with fixed-size FLOAT array for vectors
            # id: INTEGER for vector ID
            # embedding: FLOAT[dim] for the vector
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY,
                    embedding FLOAT[{dim}]
                )
                """
            )
            log.info(f"Created table {self.table_name} with {dim}-dimensional vectors")

        except Exception as e:
            log.error(f"Failed to create table {self.table_name}: {e}")
            raise e

    @contextmanager
    def init(self):
        """Create and destroy connections to database.

        For concurrent search operations, this will use read-only connections
        to avoid DuckDB file locking conflicts.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        # Use read-only connection if explicitly requested OR if we detect concurrent search mode
        # This allows multiple processes to search simultaneously without file locking conflicts
        db_config_copy = self.db_config.copy()
        if self._force_readonly:
            db_config_copy["read_only"] = True
            log.debug("Using read-only connection for concurrent access")

        self.conn = self._create_connection(**db_config_copy)
        self._setup_vss_extension()

        # Set ef_search parameter for HNSW if available
        if hasattr(self.case_config, "ef_search"):
            ef_search = self.case_config.search_param().get("ef_search", 100)
            self.conn.execute(f"SET hnsw_ef_search = {ef_search}")
            log.info(f"Set hnsw_ef_search = {ef_search}")

        try:
            yield
        finally:
            if self.conn is not None:
                self.conn.close()
                self.conn = None

    def ready_to_load(self) -> bool:
        """Check if database is ready to load data"""
        pass

    def enable_readonly_mode(self):
        """Enable read-only mode for concurrent search operations.

        This method should be called to enable concurrent searches without
        file locking conflicts. Read-only connections allow multiple processes
        to access the database simultaneously.
        """
        self._force_readonly = True
        log.info("Enabled read-only mode for concurrent search operations")

    def optimize(self, data_size: int | None = None) -> None:
        """Create index after data insertion

        For DuckDB:
        - Flat index: no index needed, brute-force search
        - HNSW index: create HNSW index on embedding column
        """
        assert self.conn is not None, "Connection is not initialized"

        index_params = self.case_config.index_param()
        index_type = index_params.get("index_type", "flat")

        if index_type == "flat":
            log.info("Using flat/brute-force search, no index created")
            return

        if index_type == "hnsw":
            metric = index_params.get("metric_type", "l2sq")
            M = index_params.get("M", 16)
            ef_construction = index_params.get("ef_construction", 128)

            log.info(
                f"Creating HNSW index on {self.table_name} with metric={metric}, "
                f"M={M}, ef_construction={ef_construction}"
            )

            try:
                # Create HNSW index on the embedding column
                # Syntax: CREATE INDEX idx_name ON table USING HNSW (column) WITH (metric = '...', ...)
                index_name = f"{self.table_name}_hnsw_idx"

                self.conn.execute(
                    f"""
                    CREATE INDEX {index_name} ON {self.table_name}
                    USING HNSW (embedding)
                    WITH (metric = '{metric}', M = {M}, ef_construction = {ef_construction})
                    """
                )

                log.info(f"Successfully created HNSW index: {index_name}")

            except Exception as e:
                log.error(f"Failed to create HNSW index: {e}")
                raise e
        else:
            log.warning(f"Unknown index type: {index_type}, skipping index creation")

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.

        Should call self.init() first.

        Args:
            embeddings: List of embedding vectors to insert
            metadata: List of IDs for the embeddings

        Returns:
            Tuple of (number of inserted records, exception if any)
        """
        assert self.conn is not None, "Connection is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            # Prepare batch data
            # DuckDB expects vectors as Python lists
            batch_data = []
            for i, row in enumerate(metadata_arr):
                vector_list = embeddings_arr[i].tolist()
                batch_data.append((int(row), vector_list))

            # Bulk insert using executemany
            # Note: Using parameterized query to avoid SQL injection
            insert_sql = f"INSERT INTO {self.table_name} (id, embedding) VALUES (?, ?)"
            self.conn.executemany(insert_sql, batch_data)

            log.debug(f"Inserted {len(metadata)} vectors into {self.table_name}")

            return len(metadata), None

        except Exception as e:
            log.error(f"Failed to insert data into {self.table_name}: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        """Search for k nearest neighbors to the query vector

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            filters: Optional filtering conditions
            timeout: Optional query timeout (not used in DuckDB)

        Returns:
            List of IDs of k nearest neighbors
        """
        assert self.conn is not None, "Connection is not initialized"

        # Format query vector as a list literal for DuckDB
        # Need to cast it to the correct FLOAT array type
        query_str = str(query)

        # Build search query using array_distance function
        # The HNSW index automatically uses the metric defined during index creation
        # array_distance() only takes 2 parameters: array_distance(vec1, vec2)
        # Syntax: SELECT id FROM table
        #         ORDER BY array_distance(embedding, query_vector)
        #         LIMIT k
        if filters:
            # Apply filters if provided
            filter_clause = f"WHERE id >= {filters.get('id', 0)}"
            search_sql = f"""
                SELECT id
                FROM {self.table_name}
                {filter_clause}
                ORDER BY array_distance(embedding, {query_str}::FLOAT[{self.dim}])
                LIMIT {k}
            """
        else:
            search_sql = f"""
                SELECT id
                FROM {self.table_name}
                ORDER BY array_distance(embedding, {query_str}::FLOAT[{self.dim}])
                LIMIT {k}
            """

        # Execute search query
        results = self.conn.execute(search_sql).fetchall()

        # Return only the IDs
        return [int(row[0]) for row in results]
