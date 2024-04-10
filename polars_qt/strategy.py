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
    fac_vol: IntoExpr | None=None,
    rev=False,
    delay_open: bool=False,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    """
    Bollinger Bands
    fac: factor to calculate bollinger bands
    params:
        if fac_vol is None
            params: window, open_width, close_width(default: 0.0), stop_width(default: None)
        if we use fac_vol stop
            params: window, open_width, close_width(default: 0.0), fac_vol_width
            the last of the params will be parsed as fac_vol_width
    min_periods: minimum periods to calculate bollinger bands
    filters: long_open, long_stop, short_open, short_stop
        for open condition, if filter is False, open behavior is disabled
        for stop condition, if filter is True, return signal will be close_signal
    fac_vol:
        a expression to calculate fac_vol, if None, we will use the default bollinger bands
    rev: reverse the long and short signal, filters will also be reversed automatically
    delay_open: if open signal is blocked by filters, whether to delay the open signal when filters are True
    """
    fac = parse_into_expr(fac)
    # process params
    params, last_param = (params[:-1], params[-1]) if fac_vol is not None else (params, None)
    if not isinstance(params, (tuple, list)):
        params = (params, 0., 0., last_param)
    elif len(params) == 1:
        params = (*params, 0., 0., last_param)
    elif len(params) == 2:
        params = (*params, 0., last_param)
    elif len(params) == 3:
        params = (*params, last_param)
    # process min_periods
    if min_periods is None:
        min_periods = params[0] // 2
    # calculate bollinger bands
    middle = fac.rolling_mean(params[0], min_periods=min_periods)
    std = fac.rolling_std(params[0], min_periods=min_periods)

    # process args and filters
    args = [fac, middle, std] if fac_vol is None else [fac, middle, std, parse_into_expr(fac_vol).cast(pl.Float64)]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filter_flag = True
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        filters = [*filters[2:], *filters[:2]] if rev else filters
        args.extend(filters)
    else:
        filter_flag = False
        args.extend([pl.lit(None)*4])
    if rev:
        long_signal, short_signal = short_signal, long_signal
    kwargs = {
        "params": params,
        "filter_flag": filter_flag,
        "delay_open": delay_open,
        "long_signal": float(long_signal),
        "short_signal": float(short_signal),
        "close_signal": float(close_signal),
    }
    return register_plugin(
        args=args,
        kwargs=kwargs,
        symbol="boll" if fac_vol is None else "boll_vol_stop",
        is_elementwise=False,
    )

