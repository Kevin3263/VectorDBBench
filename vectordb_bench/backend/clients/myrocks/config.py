from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class MyRocksConfigDict(TypedDict):
    """These keys will be directly used as kwargs in MySQL connection,
    so the names must match exactly MySQL connector API"""

    user: str
    password: str
    host: str
    port: int


class MyRocksConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 3306

    def to_dict(self) -> MyRocksConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
        }


class MyRocksIndexConfig(BaseModel):
    """Base config for MyRocks"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        # MyRocks currently only supports L2 distance via FB_VECTOR_L2
        # For now, we'll use L2 for all metrics (COSINE, IP will use L2 as fallback)
        # TODO: Add support for other distance functions when available in MyRocks
        if self.metric_type in (MetricType.L2, MetricType.COSINE, MetricType.IP, None):
            return "euclidean"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)


class MyRocksLSMConfig(MyRocksIndexConfig, DBCaseConfig):
    """Configuration for MyRocks LSM-based vector index"""

    index: IndexType = IndexType.Flat  # Using Flat as placeholder for LSM

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "lsmidx",  # MyRocks uses LSM-based index
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


_myrocks_case_config = {
    IndexType.Flat: MyRocksLSMConfig,
}
