from __future__ import annotations

from functools import wraps

import polars as pl

from .funcs import *
from .strategy import (
    auto_boll,
    auto_tangqian,
    boll,
    delay_boll,
    fix_time,
    prob_threshold,
)


@pl.api.register_expr_namespace("qt")
class ExprQuantExtend:
    def __init__(self, expr: pl.Expr):
        self.expr = expr

    def rolling_rank(self, window, min_periods=None, pct=False, rev=False) -> pl.Expr:
        return rolling_rank(
            self.expr, window=window, min_periods=min_periods, pct=pct, rev=rev
        )

    def fdiff(self, d: float, window: int, min_periods=None) -> pl.Expr:
        return fdiff(self.expr, d=d, window=window, min_periods=min_periods)

    def if_then(self, flag, then):
        return if_then(flag, then, self.expr)

    def cut(self, bins, labels, *, right=True, add_bounds=True):
        return cut(self.expr, bins, labels, right=right, add_bounds=add_bounds)

    def compose_by(self, by, method='diff'):
        return compose_by(self.expr, by, method=method)

    def half_life(self, min_periods=None) -> pl.Expr:
        return half_life(self.expr, min_periods=min_periods)

    def calc_future_ret(
        self,
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
    ):
        return calc_future_ret(
            self.expr,
            open=open,
            close=close,
            is_signal=is_signal,
            init_cash=init_cash,
            multiplier=multiplier,
            leverage=leverage,
            slippage=slippage,
            ticksize=ticksize,
            c_rate=c_rate,
            blowup=blowup,
            commision_type=commision_type,
            contract_chg_signal=contract_chg_signal,
        )

    @wraps(calc_tick_future_ret)
    def calc_tick_future_ret(self, *args, **kwargs) -> pl.Expr:
        return calc_tick_future_ret(self.expr, *args, **kwargs)

    def boll(
        self,
        params: tuple[int, float, float] | tuple[int, float] | int,
        min_periods: int | None = None,
        filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None = None,
        *,
        # fac_vol: IntoExpr | None=None,
        rev=False,
        delay_open: bool = False,
        long_signal: float = 1,
        short_signal: float = -1,
        close_signal: float = 0,
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
        rev: reverse the long and short signal, filters will also be reversed automatically
        delay_open: if open signal is blocked by filters, whether to delay the open signal when filters are True
        """
        return boll(
            self.expr,
            params=params,
            min_periods=min_periods,
            filters=filters,
            rev=rev,
            delay_open=delay_open,
            long_signal=long_signal,
            short_signal=short_signal,
            close_signal=close_signal,
    )

    def boll_rev(self, *args, **kwargs):
        return self.boll(*args, **kwargs, rev=True)


    @wraps(fix_time)
    def fix_time(self, *args, **kwargs):
        return fix_time(self.expr, *args, **kwargs)

    @wraps(prob_threshold)
    def prob_threshold(self, *args, **kwargs):
        return prob_threshold(self.expr, *args, **kwargs)

    @wraps(to_trades)
    def to_trades(self, *args, **kwargs):
        return to_trades(self.expr, *args, **kwargs)

    @wraps(auto_boll)
    def auto_boll(self, *args, **kwargs):
        return auto_boll(self.expr, *args, **kwargs)

    @wraps(auto_tangqian)
    def auto_tangqian(self, *args, **kwargs):
        return auto_tangqian(self.expr, *args, **kwargs)

    @wraps(delay_boll)
    def delay_boll(self, *args, **kwargs):
        return delay_boll(self.expr, *args, **kwargs)
