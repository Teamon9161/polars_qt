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

def fdiff(
    expr: IntoExpr, d: float, window: int, min_periods=None,
) -> pl.Expr:
    expr = parse_into_expr(expr)
    if min_periods is None:
        min_periods = window // 2
    return register_plugin(
        args=[expr],
        kwargs={
            "d": d,
            "window": window,
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
    eager: bool = False
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

def to_trades(signal: IntoExpr, time: IntoExpr, price: IntoExpr | (IntoExpr, IntoExpr)) -> pl.Expr:
    signal = parse_into_expr(signal)
    time = parse_into_expr(time)
    if isinstance(price, tuple):
        price = [parse_into_expr(p) for p in price]
    else:
        price = [parse_into_expr(price)]
    return register_plugin(
        args=[signal, time, *price],
        symbol="to_trades",
        is_elementwise=False,
    )

def compose_by(expr: IntoExpr, by: IntoExpr, method='diff') -> pl.Expr:
    expr = parse_into_expr(expr)
    if method == 'diff':
        expr = expr.diff()
    elif method is None:
        pass
    elif method == 'pct_change':
        expr = expr.pct_change()
    else:
        raise ValueError("method '{}' is not supported", method)
    by = parse_into_expr(by)
    return register_plugin(
        args=[expr, by],
        symbol="compose_by",
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
    commision_type: str = "Percent",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    open = parse_into_expr(open)
    close = parse_into_expr(close)
    signal = parse_into_expr(signal)
    pos = signal.shift(fill_value=0) if is_signal else signal
    base_config = {
        "init_cash": int(init_cash),
        "multiplier": multiplier,
        "leverage": leverage,
        "c_rate": c_rate,
        "blowup": blowup,
        "commision_type": commision_type,
    }
    assert commision_type in ["Percent", "Absolute"]
    from numbers import Number

    if isinstance(slippage, Number):
        base_config["slippage"] = slippage
        base_config["ticksize"] = ticksize
        args = [pos, open, close]
        if contract_chg_signal is not None:
            args.append(parse_into_expr(contract_chg_signal))
        return register_plugin(
            args=args,
            symbol="calc_future_ret",
            is_elementwise=False,
            kwargs=base_config,
        )
    else:
        slippage = parse_into_expr(slippage)
        args = [pos, open, close, slippage]
        if contract_chg_signal is not None:
            args.append(parse_into_expr(contract_chg_signal))
        return register_plugin(
            args=args,
            symbol="calc_future_ret_with_spread",
            is_elementwise=False,
            kwargs=base_config,
        )



def calc_tick_future_ret(
    signal: IntoExpr,
    bid: IntoExpr,
    ask: IntoExpr,
    *,
    is_signal: bool = True,
    init_cash: int = 10_000_000,
    multiplier: int = 1,
    c_rate: float = 3e-4,
    blowup: bool = False,
    commision_type: str = "Percent",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    bid = parse_into_expr(bid)
    ask = parse_into_expr(ask)
    signal = parse_into_expr(signal)
    # cast pos to signal if signal is pos
    signal = signal.shift(-1, fill_value=0) if not is_signal else signal
    kwargs = {
        "init_cash": int(init_cash),
        "multiplier": multiplier,
        "c_rate": c_rate,
        "blowup": blowup,
        "commision_type": commision_type,
    }
    assert commision_type in ["Percent", "Absolute"]

    args = [signal, bid, ask]
    if contract_chg_signal is not None:
        args.append(parse_into_expr(contract_chg_signal))
    return register_plugin(
        args=args,
        symbol="calc_tick_future_ret",
        is_elementwise=False,
        kwargs=kwargs,
    )
