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
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            required=True,
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

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Db host",
            default="127.0.0.1",
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=3306,
            help="Db Port",
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
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=MyRocksLSMConfig(),
        **parameters,
    )
