"""VectorDBBench client for lance-ivf.

Communicates with a long-running ``vdbbench_server`` Rust process via a
stdin/stdout line protocol.  One server process is started per ``init()``
context and killed on exit.

Protocol summary
----------------
INSERT / count / id f1 f2 ... fdim  (one vector per line)  → OK
OPTIMIZE                                                     → OK
SEARCH / k / count / f1 f2 ... fdim (one query per line)   → id1 id2 ... ik
QUIT                                                         → (exits)

Persistence
-----------
When ``data_path`` is set (passed via ``--data-path`` to the server), the
server saves its built index and raw-vector store to disk after OPTIMIZE.
Subsequent ``init()`` calls (in separate subprocesses) reload from disk and
serve SEARCH immediately — no re-insertion needed.

This is required because VectorDBBench runs insert, optimize, and search in
three separate ``ProcessPoolExecutor`` subprocesses, each calling ``init()``
independently.

Vectors sent to the server are already L2-normalised by the VectorDBBench
harness (because ``need_normalize_cosine`` returns True).
"""

import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..api import VectorDB
from .config import LanceIvfConfig, LanceIvfIndexConfig

log = logging.getLogger(__name__)


class LanceIvf(VectorDB):
    """lance-ivf VectorDB client."""

    name = "LanceIvf"

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: LanceIvfIndexConfig,
        collection_name: str = "lance_ivf_bench",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.binary = db_config["binary_path"]
        self.data_path = db_config.get("data_path", "/tmp/lance-ivf-vdbbench")
        self.index_cfg = db_case_config

        # Verify the binary exists
        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"vdbbench_server binary not found at {self.binary}. "
                "Run: cargo build --release --bin vdbbench_server"
            )

        if drop_old:
            self._drop_saved_state()

        # Server process is created fresh inside each init() context.
        self._proc: subprocess.Popen | None = None

    def _drop_saved_state(self):
        """Remove persisted index files so the next run starts fresh."""
        for fname in ("store.bin", "index.bin"):
            p = Path(self.data_path) / fname
            if p.exists():
                p.unlink()
                log.info(f"Removed {p}")

    def need_normalize_cosine(self) -> bool:
        """Tell the harness to L2-normalise vectors before passing them to us."""
        return True

    @contextmanager
    def init(self):
        """Start the server subprocess and yield."""
        params = self.index_cfg.index_param()
        cmd = [
            self.binary,
            "--quantizer",      params["quantizer"],
            "--num-partitions", str(params["num_partitions"]),
            "--nprobe",         str(params["nprobe"]),
            "--k-prime",        str(params["k_prime"]),
            "--dim",            str(params["dim"]),
            "--fragment-size",  str(params["fragment_size"]),
            "--data-path",      self.data_path,
        ]
        log.info(f"Starting vdbbench_server: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        try:
            yield
        finally:
            self._send_line("QUIT")
            try:
                self._proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def _send_line(self, line: str):
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_line(self) -> str:
        return self._proc.stdout.readline().rstrip("\n")

    def _expect_ok(self):
        resp = self._read_line()
        if resp != "OK":
            raise RuntimeError(f"vdbbench_server error: {resp}")

    # ------------------------------------------------------------------
    # VectorDB interface
    # ------------------------------------------------------------------

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            n = len(embeddings)
            self._send_line("INSERT")
            self._send_line(str(n))
            for emb, meta in zip(embeddings, metadata):
                vec_str = " ".join(f"{v:.8g}" for v in emb)
                self._send_line(f"{meta} {vec_str}")
            self._expect_ok()
            return n, None
        except Exception as e:
            log.warning(f"insert_embeddings error: {e}")
            return 0, e

    def optimize(self, data_size: int | None = None):
        """Compact all fragments into one IVF index and persist to data_path."""
        log.info("Sending OPTIMIZE to vdbbench_server")
        self._send_line("OPTIMIZE")
        self._expect_ok()
        log.info("OPTIMIZE complete")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Search for k nearest neighbours; returns list of original IDs."""
        self._send_line("SEARCH")
        self._send_line(str(k))
        self._send_line("1")  # single query per call
        vec_str = " ".join(f"{v:.8g}" for v in query)
        self._send_line(vec_str)

        result_line = self._read_line()
        if result_line.startswith("ERROR"):
            raise RuntimeError(f"vdbbench_server search error: {result_line}")

        if not result_line.strip():
            return []
        return [int(x) for x in result_line.split()]
