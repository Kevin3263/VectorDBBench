"""Click commands for the lance-ivf VectorDBBench client."""

from typing import Annotated, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import IndexType


class LanceIvfTypedDict(CommonTypedDict):
    binary_path: Annotated[
        str,
        click.option(
            "--binary-path",
            type=str,
            default="~/lance-ivf/target/release/vdbbench_server",
            help="Path to the vdbbench_server binary",
        ),
    ]
    data_path: Annotated[
        str,
        click.option(
            "--data-path",
            type=str,
            default="/tmp/lance-ivf-vdbbench",
            help="Directory for cross-process index persistence",
        ),
    ]
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=1024,
            help="Number of IVF partitions",
        ),
    ]
    nprobe: Annotated[
        int,
        click.option(
            "--nprobe",
            type=int,
            default=64,
            help="Number of partitions to probe per query",
        ),
    ]
    k_prime: Annotated[
        int,
        click.option(
            "--k-prime",
            type=int,
            default=0,
            help="Reranking over-fetch size (0 = disabled)",
        ),
    ]
    dim: Annotated[
        int,
        click.option(
            "--dim",
            type=int,
            default=768,
            help="Vector dimension",
        ),
    ]
    fragment_size: Annotated[
        int,
        click.option(
            "--fragment-size",
            type=int,
            default=50000,
            help="Buffer this many vectors before flushing a fragment",
        ),
    ]


def _make_run(quantizer: str):
    """Return a click command runner for the given quantizer mode."""

    def _run(**parameters: Unpack[LanceIvfTypedDict]):
        from .config import LanceIvfConfig, LanceIvfIndexConfig

        binary = parameters["binary_path"].replace("~", str(__import__("pathlib").Path.home()))
        run(
            db=DB.LanceIvf,
            db_config=LanceIvfConfig(
                db_label=parameters["db_label"],
                binary_path=binary,
                data_path=parameters["data_path"],
            ),
            db_case_config=LanceIvfIndexConfig(
                quantizer=quantizer,
                num_partitions=parameters["num_partitions"],
                nprobe=parameters["nprobe"],
                k_prime=parameters["k_prime"],
                dim=parameters["dim"],
                fragment_size=parameters["fragment_size"],
            ),
            **parameters,
        )

    return _run


@cli.command("LanceIvfFlat")
@click_parameter_decorators_from_typed_dict(LanceIvfTypedDict)
def LanceIvfFlat(**parameters: Unpack[LanceIvfTypedDict]):
    """lance-ivf with flat (no quantization) IVF index."""
    _make_run("flat")(**parameters)


@cli.command("LanceIvfSQ8")
@click_parameter_decorators_from_typed_dict(LanceIvfTypedDict)
def LanceIvfSQ8(**parameters: Unpack[LanceIvfTypedDict]):
    """lance-ivf with SQ8 (8-bit scalar quantization) IVF index."""
    _make_run("sq8")(**parameters)


@cli.command("LanceIvfRaBitQ")
@click_parameter_decorators_from_typed_dict(LanceIvfTypedDict)
def LanceIvfRaBitQ(**parameters: Unpack[LanceIvfTypedDict]):
    """lance-ivf with RaBitQ IVF index + reranking."""
    _make_run("rabitq")(**parameters)
