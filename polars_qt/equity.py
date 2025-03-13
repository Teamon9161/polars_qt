from __future__ import annotations

from typing import TYPE_CHECKING

from polars_qt.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    import polars as pl
    from polars.type_aliases import IntoExpr


def to_trades(
    signal: IntoExpr,
    time: IntoExpr,
    price: IntoExpr | tuple[IntoExpr, IntoExpr],
) -> pl.Expr:
    signal = parse_into_expr(signal)
    time = parse_into_expr(time)
    if isinstance(price, (tuple, list)):
        price = [parse_into_expr(p) for p in price]
    else:
        price = [parse_into_expr(price)]
    return register_plugin(
        args=[signal, time, *price],
        symbol="to_trades",
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
    c_rate: float = 3e-4,
    blowup: bool = False,
    commission_type: str = "Percent",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    """
    Calculate future return.
    signal:
        signal to trade, 1 for long, -1 for short, 0 for close position, 0.5 for half long position
    open: open price series, this will be used as open price when trading
    close: close price series, this will be used to calculate return
    is_signal: signal series is signal or position series, position series is signal series shift 1
    init_cash: initial cash
    multiplier: contract multiplier
    leverage: leverage(deprecated, will be removed in future version)
    slippage: slippage, can be a float or a series
    c_rate: commission rate
    blowup: whether to blow up account when cash is less than 0
    commission_type: commission type, Percent or Absolute
        percent: percent | pct
        absolute: absolute | fixed | fix
    signal_type: signal type, Percent or Absolute
        percent: percent | pct
        absolute: absolute | fixed | fix
    contract_chg_signal: signal to change contract, series of boolean dtype
    """
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
        "commission_type": commission_type,
    }
    from numbers import Number

    if isinstance(slippage, Number):
        base_config["slippage"] = slippage
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
    commission_type: str = "Percent",
    signal_type: str = "Percent",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    """
    Calculate future return with tick data.
    signal:
        if signal_type is percent:
            signal to trade, 1 for long, -1 for short, 0 for close position, 0.5 for half long position
        if signal_type is absolute:
            lot_num signal to trade
    bid: bid1 price series
    ask: ask1 price series
    is_signal: signal series is signal or position series, position series is signal series shift 1
    init_cash: initial cash
    multiplier: contract multiplier
    c_rate: commission rate
    blowup: whether to blow up account when cash is less than 0
    commission_type: commission type, Percent or Absolute
        percent: percent | pct
        absolute: absolute | fixed | fix
    signal_type: signal type, Percent or Absolute
        percent: percent | pct
        absolute: absolute | fixed | fix
    contract_chg_signal: signal to change contract, series of boolean dtype
    """
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
        "commission_type": commission_type,
        "signal_type": signal_type,
    }

    args = [signal, bid, ask]
    if contract_chg_signal is not None:
        args.append(parse_into_expr(contract_chg_signal))
    return register_plugin(
        args=args,
        symbol="calc_tick_future_ret",
        is_elementwise=False,
        kwargs=kwargs,
    )


def calc_tick_future_ret_full(
    signal: IntoExpr,
    bid: IntoExpr,
    ask: IntoExpr,
    *,
    is_signal: bool = True,
    init_cash: int = 0,
    multiplier: int = 1,
    c_rate: float = 3e-4,
    blowup: bool = False,
    commission_type: str = "Percent",
    # signal_type: str = "Absolute",
    open_price_method: str = "average",
    contract_chg_signal: IntoExpr | None = None,
) -> pl.Expr:
    """
    Calculate future return with tick data.
    signal:
        if signal_type is percent:
            signal to trade, 1 for long, -1 for short, 0 for close position, 0.5 for half long position
        if signal_type is absolute:
            lot_num signal to trade
    bid: bid1 price series
    ask: ask1 price series
    is_signal: signal series is signal or position series, position series is signal series shift 1
    init_cash: initial cash
    multiplier: contract multiplier
    c_rate: commission rate
    blowup: whether to blow up account when cash is less than 0
    commission_type: commission type, Percent or Absolute
        percent: percent | pct
        absolute: absolute | fixed | fix
    open_price_method: first | last | average
    contract_chg_signal: signal to change contract, series of boolean dtype
    """
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
        "commission_type": commission_type,
        "signal_type": "absolute",
        "open_price_method": open_price_method,
    }

    args = [signal, bid, ask]
    if contract_chg_signal is not None:
        args.append(parse_into_expr(contract_chg_signal))
    return register_plugin(
        args=args,
        symbol="calc_tick_future_ret_full",
        is_elementwise=False,
        kwargs=kwargs,
    )
