import os
from pathlib import Path

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class LanceIvfConfig(DBConfig):
    """Connection config for lance-ivf.

    The only required field is ``binary_path``: the path to the
    ``vdbbench_server`` release binary.
    """

    db_label: str = "lance-ivf"
    binary_path: str = str(
        Path.home() / "lance-ivf" / "target" / "release" / "vdbbench_server"
    )
    data_path: str = "/tmp/lance-ivf-vdbbench"

    def to_dict(self) -> dict:
        return {"binary_path": self.binary_path, "data_path": self.data_path}


class LanceIvfIndexConfig(BaseModel, DBCaseConfig):
    """Index and search parameters for a lance-ivf run."""

    index: IndexType = IndexType.IVFFlat
    metric_type: MetricType = MetricType.COSINE
    quantizer: str = "flat"          # flat | sq8 | rabitq
    num_partitions: int = 1024
    nprobe: int = 64
    k_prime: int = 0                 # >0 enables reranking
    dim: int = 768
    fragment_size: int = 50_000      # buffer this many vectors before flushing a fragment

    def index_param(self) -> dict:
        return {
            "quantizer": self.quantizer,
            "num_partitions": self.num_partitions,
            "nprobe": self.nprobe,
            "k_prime": self.k_prime,
            "dim": self.dim,
            "fragment_size": self.fragment_size,
        }

    def search_param(self) -> dict:
        return {}


_lance_ivf_case_config = {
    IndexType.IVFFlat: LanceIvfIndexConfig,
}
