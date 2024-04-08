import polars as pl
from polars.testing import assert_series_equal

import polars_qt as pq


def test_if_then():
    # test basic
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        }
    )
    res = df.select(c=pl.col("a").qt.if_then(pl.col("a").sum() > 13, "b"))
    assert_series_equal(res["c"], df["b"], check_names=False)
    df[4, "a"] = 0
    res = df.select(c=pl.col("a").qt.if_then(pl.col("a").sum() > 13, "b"))
    assert_series_equal(res["c"], df["a"], check_names=False)

    # test in group_by context
    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "a", "b"],
            "v": [1, 3, 5, 2, 4],
        }
    )
    res = df.select(pl.col("v").qt.if_then((pl.len() > 2), pl.col("v") * 2).over("g"))
    assert_series_equal(res["v"], pl.Series([2, 6, 5, 4, 4]), check_names=False)
