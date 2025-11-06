from typing import TypedDict

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class DuckDBConfigDict(TypedDict):
    """Configuration dictionary for DuckDB connection"""

    database: str
    read_only: bool


class DuckDBConfig(DBConfig):
    """DuckDB connection configuration

    Args:
        database: Database file path (or ':memory:' for in-memory database)
        read_only: Whether to open database in read-only mode
        threads: Number of threads for parallel operations (None = auto)
    """

    database: str = ":memory:"
    read_only: bool = False
    threads: int | None = None

    def to_dict(self) -> DuckDBConfigDict:
        return {
            "database": self.database,
            "read_only": self.read_only,
        }


class DuckDBIndexConfig(BaseModel):
    """Base configuration for DuckDB indexes"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        """Map MetricType to DuckDB VSS metric string

        DuckDB VSS supports:
        - 'l2sq' for L2 squared distance
        - 'cosine' for cosine distance
        - 'ip' for inner product
        """
        if self.metric_type == MetricType.L2:
            return "l2sq"
        elif self.metric_type == MetricType.COSINE:
            return "cosine"
        elif self.metric_type == MetricType.IP:
            return "ip"
        elif self.metric_type is None:
            # Default to L2
            return "l2sq"
        else:
            msg = f"Metric type {self.metric_type} is not supported by DuckDB!"
            raise ValueError(msg)


class DuckDBFlatConfig(DuckDBIndexConfig, DBCaseConfig):
    """Configuration for DuckDB without index (flat/brute-force search)"""

    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "flat",
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


class DuckDBHNSWConfig(DuckDBIndexConfig, DBCaseConfig):
    """Configuration for DuckDB HNSW index

    Args:
        M: Number of bi-directional links per node (default: 16)
            Higher M = better recall, more memory
            Typical range: 8-64
        ef_construction: Size of dynamic candidate list during index build (default: 128)
            Higher ef_construction = better recall, slower indexing
            Typical range: 64-512
        ef_search: Size of dynamic candidate list during search (default: 100)
            Higher ef_search = better recall, slower search
            Typical range: 50-500
    """

    index: IndexType = IndexType.HNSW
    M: int = 16
    ef_construction: int = 128
    ef_search: int = 100

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "hnsw",
            "M": self.M,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "ef_search": self.ef_search,
        }


_duckdb_case_config = {
    IndexType.Flat: DuckDBFlatConfig,
    IndexType.HNSW: DuckDBHNSWConfig,
}
