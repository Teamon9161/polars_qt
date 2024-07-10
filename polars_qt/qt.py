from __future__ import annotations

from functools import wraps

import polars as pl

from .equity import (
    calc_future_ret,
    calc_tick_future_ret,
    to_trades,
)
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

    def calc_future_ret(self, *args, **kwargs) -> pl.Expr:
        return calc_future_ret(self.expr, *args, **kwargs)

    def calc_tick_future_ret(self, *args, **kwargs) -> pl.Expr:
        return calc_tick_future_ret(self.expr, *args, **kwargs)

    def boll(self, *args, **kwargs) -> pl.Expr:
        return boll(self.expr, *args, **kwargs)

    def boll_rev(self, *args, **kwargs):
        return self.boll(*args, **kwargs, rev=True)

    def fix_time(self, *args, **kwargs):
        return fix_time(self.expr, *args, **kwargs)

    def prob_threshold(self, *args, **kwargs):
        return prob_threshold(self.expr, *args, **kwargs)

    def to_trades(self, *args, **kwargs):
        return to_trades(self.expr, *args, **kwargs)

    def auto_boll(self, *args, **kwargs):
        return auto_boll(self.expr, *args, **kwargs)

    def auto_tangqian(self, *args, **kwargs):
        return auto_tangqian(self.expr, *args, **kwargs)

    def delay_boll(self, *args, **kwargs):
        return delay_boll(self.expr, *args, **kwargs)

    def rolling_ewm(self, *args, **kwargs):
        return rolling_ewm(self.expr, *args, **kwargs)

    def tick_up_prob(self, *args, **kwargs):
        return tick_up_prob(self.expr, *args, **kwargs)
