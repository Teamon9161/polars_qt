from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr, PolarsDataType


def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    # Simple version parser; split into a tuple of ints for comparison.
    # vendored from Polars
    if isinstance(version, str):
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)


OLD_POLARS = parse_version(pl.__version__) < parse_version("0.20.16")

if OLD_POLARS:
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent


def parse_into_expr(
    expr: IntoExpr,
    *,
    str_as_lit: bool = False,
    list_as_lit: bool = True,
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    expr
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to `False` (default),
        strings are parsed as column names.
    list_as_lit
        Interpret list input as a lit literal, If set to `False`,
        lists are parsed as `Series` literals.
    dtype
        If the input is expected to resolve to a literal with a known dtype, pass
        this to the `lit` constructor.

    Returns
    -------
    polars.Expr
    """
    if isinstance(expr, pl.Expr):
        pass
    elif isinstance(expr, str) and not str_as_lit:
        expr = pl.col(expr)
    elif isinstance(expr, list) and not list_as_lit:
        expr = pl.lit(pl.Series(expr), dtype=dtype)
    else:
        expr = pl.lit(expr, dtype=dtype)

    return expr


def register_plugin(
    *,
    symbol: str,
    is_elementwise: bool,
    kwargs: dict[str, Any] | None = None,
    args: list[IntoExpr],
    # lib: str | Path,
) -> pl.Expr:
    # global lib
    if OLD_POLARS:
        assert isinstance(args[0], pl.Expr)
        assert isinstance(lib, str)
        return args[0].register_plugin(
            lib=lib,
            symbol=symbol,
            args=args[1:],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
        )
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        args=args,
        plugin_path=lib,
        function_name=symbol,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
    )
