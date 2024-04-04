import polars as pl
from .funcs import *

@pl.api.register_expr_namespace("qt")
class ExprQuantExtend:
    def __init__(self, expr: pl.Expr):
        self.expr = expr

    def rolling_rank(self, window, min_periods=None, pct=False, rev=False) -> pl.Expr:
        return rolling_rank(self.expr, window=window, min_periods=min_periods, pct=pct, rev=rev)
    