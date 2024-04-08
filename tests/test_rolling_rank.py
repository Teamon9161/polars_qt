import polars as pl
from polars.testing import assert_series_equal

import polars_qt as pq


def test_rolling_rank():
    df = pl.DataFrame(
        {
            "a": [5.2, 4.1, 6.3, None, 10, 4, 5],
        }
    )
    df = df.with_columns(
        pq.rolling_rank(pl.col("a"), 4, min_periods=1, pct=True).alias("a_rank"),
        pl.col("a").qt.rolling_rank(4, pct=False, rev=True).alias("a_rank2"),
    )
    assert_series_equal(
        df["a_rank"],
        pl.Series([1, 0.5, 1.0, None, 1.0, 1 / 3, 2 / 3]),
        check_names=False,
    )
    assert_series_equal(
        df["a_rank2"], pl.Series([None, 2.0, 1, None, 1, 3, 2]), check_names=False
    )
