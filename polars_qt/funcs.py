
from typing import TYPE_CHECKING

import polars as pl
from polars.type_aliases import IntoExpr

from polars_qt.utils import parse_into_expr, parse_version, register_plugin


def rolling_rank(expr: IntoExpr, window, min_periods=None, pct=False, rev=False) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            'window': window,
            "min_periods": min_periods,
            "pct": pct,
            "rev": rev,
        },
        symbol="rolling_rank",
        is_elementwise=False,
    )