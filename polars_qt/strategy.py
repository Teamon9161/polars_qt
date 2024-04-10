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
    delay_open: bool=False,
    long_signal: float=1,
    short_signal: float=-1,
    close_signal: float=0,
) -> pl.Expr:
    fac = parse_into_expr(fac)
    if not isinstance(params, (tuple, list)):
        params = (params, 0., 0., None)
    elif len(params) == 2:
        params = (*params, 0., None)
    elif len(params) == 3:
        params = (*params, None)
    if min_periods is None:
        min_periods = params[0] // 2
    middle = fac.rolling_mean(params[0], min_periods=min_periods)
    std = fac.rolling_std(params[0], min_periods=min_periods)
    args = [fac, middle, std]
    if filters is not None:
        assert len(filters) == 4, "filters must be a list of 4 elements"
        filter_flag = True
        filters = [parse_into_expr(f).cast(pl.Boolean) if not isinstance(f, bool) else pl.repeat(f, fac.len()) for f in filters]
        args.extend(filters)
    else:
        filter_flag = False
        args.extend([pl.lit(None)*4])
    return register_plugin(
        args=args,
        kwargs={
            "params": params,
            "filter_flag": filter_flag,
            "delay_open": delay_open,
            "long_signal": float(long_signal),
            "short_signal": float(short_signal),
            "close_signal": float(close_signal),
        },
        symbol="boll",
        is_elementwise=False,
    )
