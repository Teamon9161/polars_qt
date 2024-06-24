from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_qt.utils import parse_into_expr, register_plugin

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


def boll(
    fac: IntoExpr,
    params: tuple[int, float, float] | tuple[int, float] | int,
    min_periods: int | None=None,
    filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None=None,
    *,
    # fac_vol: IntoExpr | None=None,
    rev=False,
    delay_open: bool=True,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    """
    Bollinger Bands
    fac: factor to calculate bollinger bands
    params:
        params: window, open_width, stop_width(default: 0.0), take_profit_width(default: None)
    min_periods: minimum periods to calculate bollinger bands
    filters: long_open, long_stop, short_open, short_stop
        for open condition, if filter is False, open behavior is disabled
        for stop condition, if filter is True, return signal will be close_signal
    rev: reverse the long and short signal, filters will also be reversed automatically
    delay_open: if open signal is blocked by filters, whether to delay the open signal when filters are True
    """
    fac = parse_into_expr(fac)
    # process params
    params, last_param = (params, None)
    if not isinstance(params, (tuple, list)):
        params = (params, 0., 0., last_param)
    elif len(params) == 1:
        params = (*params, 0., 0., last_param)
    elif len(params) == 2:
        params = (*params, 0., last_param)
    elif len(params) == 3:
        params = (*params, last_param)

    # process args and filters
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        filters = [*filters[2:], *filters[:2]] if rev else filters
        args.extend(filters)
    else:
        pass
    if rev:
        long_signal, short_signal = short_signal, long_signal
    kwargs = {
        "params": params,
        "min_periods": min_periods,
        "delay_open": delay_open,
        "long_signal": float(long_signal),
        "short_signal": float(short_signal),
        "close_signal": float(close_signal),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol="boll",
        is_elementwise=False,
    )

def auto_boll(
    fac: IntoExpr,
    params: tuple[int, float, float] | tuple[int, float] | int,
    pos_map: (list[float], list[float]) | None=None,
    min_periods: int | None=None,
    filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None=None,
    *,
    rev=False,
    delay_open: bool=True,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    """
    Bollinger Bands
    fac: factor to calculate bollinger bands
    params:
        params: window, open_width, stop_width(default: 0.0)
    min_periods: minimum periods to calculate bollinger bands
    filters: long_open, long_stop, short_open, short_stop
        for open condition, if filter is False, open behavior is disabled
        for stop condition, if filter is True, return signal will be close_signal
    rev: reverse the long and short signal, filters will also be reversed automatically
    delay_open: if open signal is blocked by filters, whether to delay the open signal when filters are True
    """
    fac = parse_into_expr(fac)
    # process params
    if not isinstance(params, (tuple, list)):
        params = (params, 0., 0.)
    elif len(params) == 1:
        params = (*params, 0., 0.)
    elif len(params) == 2:
        params = (*params, 0.)

    # process args and filters
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        filters = [*filters[2:], *filters[:2]] if rev else filters
        args.extend(filters)
    if rev:
        long_signal, short_signal = short_signal, long_signal

    kwargs = {
        "params": params,
        "min_periods": min_periods,
        "pos_map": pos_map,
        "delay_open": delay_open,
        "long_signal": float(long_signal),
        "short_signal": float(short_signal),
        "close_signal": float(close_signal),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol="auto_boll",
        is_elementwise=False,
    )

def delay_boll(
    fac: IntoExpr,
    params: tuple[int, float, float, float] | tuple[int, float, float],
    chase_bound: float | None=None,
    min_periods: int | None=None,
    filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None=None,
    *,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    """
    Bollinger Bands but only open when fac fall back.
    fac: factor to calculate bollinger bands
    params:
        params: window, open_width, stop_width(default: 0.0), delay_open_width
        the last of the params will always be parsed as delay_open_width
    chase_bound: whether to chase high if fac doesn't fall back
    min_periods: minimum periods to calculate bollinger bands
    filters: long_open, long_stop, short_open, short_stop
        for open condition, if filter is False, open behavior is disabled
        for stop condition, if filter is True, return signal will be close_signal
    """
    fac = parse_into_expr(fac)
    # process params
    params, last_param = (params[:-1], params[-1])
    if len(params) == 2:
        params = (*params, 0., last_param)
    elif len(params) == 3:
        params = (*params, last_param)
    params = (*params, chase_bound)
    # process args and filters
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        args.extend(filters)
    else:
        pass
    kwargs = {
        "params": params,
        "min_periods": min_periods,
        "long_signal": float(long_signal),
        "short_signal": float(short_signal),
        "close_signal": float(close_signal),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol="delay_boll",
        is_elementwise=False,
    )

def martingale(
    close: IntoExpr,
    n: int,
    step: int | None,
    init_pos: float,
    take_profit: float,
    filters: IntoExpr | None=None,
    win_p_addup: float | None = None,
    pos_mul: float | None = 2,
    b: float=1,
    stop_loss_m: float | None = None,
) -> pl.Expr:
    close = parse_into_expr(close)
    args = [close]
    if filters is not None:
        args.extend([parse_into_expr(filters).cast(pl.Boolean), pl.repeat(True, close.len()), pl.repeat(False, close.len()), pl.repeat(False, close.len())])
    else:
        pass
    kwargs = {
        'n': n,
        'step': step,
        'init_pos': init_pos,
        'win_p_addup': win_p_addup,
        'pos_mul': pos_mul,
        'take_profit': take_profit,
        'b': b,
        'stop_loss_m': stop_loss_m,
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol='martingale',
        is_elementwise=False,
    )

def fix_time(
    fac: IntoExpr,
    n: int,
    pos_map: tuple(list) | None,
    filters: IntoExpr | None=None,
    *,
    extend_time: bool=False,
) -> pl.Expr:
    fac = parse_into_expr(fac)
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        args.extend(filters)
    kwargs = {
        'n': n,
        'pos_map': pos_map,
        'extend_time': extend_time,
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol='fix_time',
        is_elementwise=False,
    )

def auto_tangqian(
    fac: IntoExpr,
    params: tuple[int, float, float] | tuple[int, float] | int,
    pos_map: (list[float], list[float]) | None=None,
    min_periods: int | None=None,
    filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None=None,
    *,
    rev=False,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    """
    TangQian Bands Strategy
    fac: factor to calculate TangQiAn bands
    params:
        params: window, open_width, stop_width(default: 0.0)
    min_periods: minimum periods to calculate bollinger bands
    filters: long_open, long_stop, short_open, short_stop
        for open condition, if filter is False, open behavior is disabled
        for stop condition, if filter is True, return signal will be close_signal
    rev: reverse the long and short signal, filters will also be reversed automatically
    delay_open: if open signal is blocked by filters, whether to delay the open signal when filters are True
    """
    fac = parse_into_expr(fac)
    # process params
    if not isinstance(params, (tuple, list)):
        params = (params, 0., 0.)
    elif len(params) == 1:
        params = (*params, 0., 0.)
    elif len(params) == 2:
        params = (*params, 0.)

    # process args and filters
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        filters = [*filters[2:], *filters[:2]] if rev else filters
        args.extend(filters)
    if rev:
        long_signal, short_signal = short_signal, long_signal

    kwargs = {
        "params": params,
        "min_periods": min_periods,
        "pos_map": pos_map,
        "long_signal": float(long_signal),
        "short_signal": float(short_signal),
        "close_signal": float(close_signal),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol="auto_tangqian",
        is_elementwise=False,
    )

def prob_threshold(
    fac: IntoExpr,
    thresholds: (float, float, float, float),
    per_hand = 1.,
    max_hand = 3.,
    filters: IntoExpr | None=None,
) -> pl.Expr:
    fac = parse_into_expr(fac)
    args = [fac]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        args.extend(filters)
    kwargs = {
        'thresholds': thresholds,
        'per_hand': float(per_hand),
        'max_hand': float(max_hand),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol='prob_threshold',
        is_elementwise=False,
    )
