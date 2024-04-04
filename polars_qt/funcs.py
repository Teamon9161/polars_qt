import polars as pl
from polars.type_aliases import IntoExpr

from polars_qt.utils import parse_into_expr, register_plugin


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
    

def if_then(flag_expr: IntoExpr, expr1: IntoExpr, expr2: IntoExpr) -> pl.Expr:
    flag_expr = parse_into_expr(flag_expr) if not isinstance(flag_expr, bool) else pl.lit(flag_expr)
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[flag_expr, expr1, expr2],
        symbol="if_then",
        is_elementwise=False,
    )