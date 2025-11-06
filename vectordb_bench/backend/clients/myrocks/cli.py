from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class MyRocksTypedDict(CommonTypedDict):
    username: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            default="root",
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            required=True,
        ),
    ]
    nprobe: Annotated[
        int,
        click.option(
            "--nprobe",
            type=int,
            help="Number of nearest centroids to search (1-10000, default: 16)",
            default=16,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(MyRocksTypedDict)
def MyRocks(
    **parameters: Unpack[MyRocksTypedDict],
):
    from .config import MyRocksConfig, MyRocksLSMConfig

    run(
        db=DB.MyRocks,
        db_config=MyRocksConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
        ),
        db_case_config=MyRocksLSMConfig(
            nprobe=parameters["nprobe"],
        ),
        **parameters,
    )
