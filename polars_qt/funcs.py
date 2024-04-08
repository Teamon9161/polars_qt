from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_qt.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
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


def calc_future_ret(
    signal: IntoExpr,
    open: IntoExpr,
    close: IntoExpr,
    *,
    is_signal: bool = True,
    init_cash: int = 10_000_000,
    multiplier: int = 1,
    leverage: float = 1,
    slippage: float | IntoExpr = 0,
    ticksize: float = 0,
    c_rate: float = 3e-4,
    blowup: bool = False,
    commision_type: str = "percent",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    open = parse_into_expr(open).cast(pl.Float64)
    close = parse_into_expr(close).cast(pl.Float64)
    signal = parse_into_expr(signal).cast(pl.Float64)
    pos = signal.shift(fill_value=0) if is_signal else signal
    base_config = {
        "init_cash": init_cash,
        "multiplier": multiplier,
        "leverage": leverage,
        "c_rate": c_rate,
        "blowup": blowup,
        "commision_type": commision_type,
    }
    assert commision_type in ["percent", "absolute"]
    from numbers import Number

    if isinstance(slippage, Number):
        base_config["slippage"] = slippage
        base_config["ticksize"] = ticksize
        args = [pos, open, close]
        if contract_chg_signal is not None:
            args.append(parse_into_expr(contract_chg_signal).cast(pl.Boolean))
        return register_plugin(
            args=args,
            symbol="calc_future_ret",
            is_elementwise=False,
            kwargs=base_config,
        )
    else:
        slippage = parse_into_expr(slippage).cast(pl.Float64)
        args = [pos, open, close, slippage]
        if contract_chg_signal is not None:
            args.append(parse_into_expr(contract_chg_signal).cast(pl.Boolean))
        return register_plugin(
            args=args,
            symbol="calc_future_ret_with_spread",
            is_elementwise=False,
            kwargs=base_config,
        )
