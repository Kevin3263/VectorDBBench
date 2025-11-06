from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class DuckDBTypedDict(CommonTypedDict):
    database: Annotated[
        str,
        click.option(
            "--database",
            type=str,
            help="Database file path (use ':memory:' for in-memory database)",
            default=":memory:",
        ),
    ]
    threads: Annotated[
        int | None,
        click.option(
            "--threads",
            type=int,
            help="Number of threads for parallel operations (default: auto)",
            default=None,
        ),
    ]
    hnsw_m: Annotated[
        int,
        click.option(
            "--hnsw-m",
            type=int,
            help="HNSW M parameter: number of bi-directional links per node (8-64, default: 16)",
            default=16,
        ),
    ]
    hnsw_ef_construction: Annotated[
        int,
        click.option(
            "--hnsw-ef-construction",
            type=int,
            help="HNSW ef_construction: size of dynamic candidate list during build (64-512, default: 128)",
            default=128,
        ),
    ]
    hnsw_ef_search: Annotated[
        int,
        click.option(
            "--hnsw-ef-search",
            type=int,
            help="HNSW ef_search: size of dynamic candidate list during search (50-500, default: 100)",
            default=100,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBTypedDict)
def DuckDB(
    **parameters: Unpack[DuckDBTypedDict],
):
    from .config import DuckDBConfig, DuckDBHNSWConfig

    run(
        db=DB.DuckDB,
        db_config=DuckDBConfig(
            db_label=parameters["db_label"],
            database=parameters["database"],
            threads=parameters["threads"],
        ),
        db_case_config=DuckDBHNSWConfig(
            M=parameters["hnsw_m"],
            ef_construction=parameters["hnsw_ef_construction"],
            ef_search=parameters["hnsw_ef_search"],
        ),
        **parameters,
    )
