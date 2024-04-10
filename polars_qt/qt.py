from __future__ import annotations

import polars as pl

from .funcs import *
from .strategy import boll


@pl.api.register_expr_namespace("qt")
class ExprQuantExtend:
    def __init__(self, expr: pl.Expr):
        self.expr = expr

    def rolling_rank(self, window, min_periods=None, pct=False, rev=False) -> pl.Expr:
        return rolling_rank(
            self.expr, window=window, min_periods=min_periods, pct=pct, rev=rev
        )

    def if_then(self, flag, then):
        return if_then(flag, then, self.expr)

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
        commision_type: str = "percent",
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

    def boll(
        self,
        params: tuple[int, float, float] | tuple[int, float] | int,
        min_periods: int | None = None,
        filters: tuple[IntoExpr, IntoExpr, IntoExpr, IntoExpr] | None = None,
        *,
        delay_open: bool = False,
        long_signal: float = 1,
        short_signal: float = -1,
        close_signal: float = 0,
    ) -> pl.Expr:
        return boll(
            self.expr,
            params=params,
            min_periods=min_periods,
            filters=filters,
            delay_open=delay_open,
            long_signal=long_signal,
            short_signal=short_signal,
            close_signal=close_signal,
    )
