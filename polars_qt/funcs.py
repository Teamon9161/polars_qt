from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_qt.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from datetime import datetime

    from polars.type_aliases import IntoExpr


def rolling_rank(
    expr: IntoExpr, window, min_periods=None, pct=False, rev=False
) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "window": window,
            "min_periods": min_periods,
            "pct": pct,
            "rev": rev,
        },
        symbol="rolling_rank",
        is_elementwise=False,
    )


def rolling_kurt(expr: IntoExpr, window, min_periods=None) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "window": window,
            "min_periods": min_periods,
        },
        symbol="rolling_kurt",
        is_elementwise=False,
    )


def rolling_zscore(expr: IntoExpr, window, min_periods=None) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "window": window,
            "min_periods": min_periods,
        },
        symbol="rolling_zscore",
        is_elementwise=False,
    )


def zscore(expr: IntoExpr, min_periods=None) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin(
        args=[expr],
        kwargs={
            "min_periods": min_periods,
        },
        symbol="zscore",
        is_elementwise=False,
    )


def tick_up_prob(n_ask: IntoExpr, n_bid: IntoExpr, degree=None) -> pl.Expr:
    n_ask = parse_into_expr(n_ask)
    n_bid = parse_into_expr(n_bid)
    return register_plugin(
        args=[n_ask, n_bid],
        kwargs={
            "degree": degree,
        },
        symbol="tick_up_prob",
        is_elementwise=True,
    )


def rolling_ewm(expr: IntoExpr, window, min_periods=None) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "window": window,
            "min_periods": min_periods,
        },
        symbol="rolling_ewm",
        is_elementwise=False,
    )


def fdiff(
    expr: IntoExpr,
    d: float,
    window: int,
    ignore_na=True,
    min_periods=None,
) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "d": d,
            "window": window,
            "ignore_na": ignore_na,
            "min_periods": min_periods,
        },
        symbol="fdiff",
        is_elementwise=False,
    )


def linspace(start: IntoExpr, stop: IntoExpr, num: IntoExpr, eager=True) -> pl.Expr:
    start = parse_into_expr(start)
    stop = parse_into_expr(stop)
    num = parse_into_expr(num)
    res = register_plugin(
        args=[start, stop, num],
        symbol="linspace",
        is_elementwise=False,
    )
    if eager:
        return pl.select(res).to_series()
    else:
        return res


def cut(
    fac: IntoExpr,
    bins: IntoExpr,
    labels: IntoExpr,
    *,
    right: bool = True,
    add_bounds: bool = True,
    eager: bool = False,
) -> pl.Expr:
    fac = parse_into_expr(fac, list_as_lit=False)
    bins = parse_into_expr(bins, list_as_lit=False)
    labels = parse_into_expr(labels, list_as_lit=False)
    res = register_plugin(
        args=[fac, bins, labels],
        kwargs={"right": right, "add_bounds": add_bounds},
        symbol="cut",
        is_elementwise=True,
    )
    if eager:
        return pl.select(res).to_series()
    else:
        return res


def if_then(flag_expr: IntoExpr, expr1: IntoExpr, expr2: IntoExpr) -> pl.Expr:
    flag_expr = (
        parse_into_expr(flag_expr)
        if not isinstance(flag_expr, bool)
        else pl.lit(flag_expr)
    )
    expr1 = parse_into_expr(expr1)
    expr2 = parse_into_expr(expr2)
    return register_plugin(
        args=[flag_expr, expr1, expr2],
        symbol="if_then",
        is_elementwise=False,
    )


def half_life(fac: IntoExpr, min_periods=None) -> pl.Expr:
    fac = parse_into_expr(fac)
    return register_plugin(
        args=[fac],
        symbol="half_life",
        kwargs={"min_periods": min_periods},
        is_elementwise=False,
    )


def compose_by(expr: IntoExpr, by: IntoExpr, method="diff") -> pl.Expr:
    expr = parse_into_expr(expr)
    if method == "diff":
        expr = expr.diff()
    elif method is None:
        pass
    elif method == "pct_change":
        expr = expr.pct_change()
    else:
        raise ValueError("method '{}' is not supported", method)
    by = parse_into_expr(by)
    return register_plugin(
        args=[expr, by],
        symbol="compose_by",
        is_elementwise=False,
    )


def to_datetime(t: str | datetime) -> datetime:
    if isinstance(t, str):
        return pl.Series([t]).str.to_datetime()[0]
    else:
        return t
