import numpy as np
import polars as pl
from numpy.testing import assert_allclose
from polars.testing import assert_series_equal

import polars_qt as pq


def test_compose_by():
    df = pl.DataFrame({
        'p': [100, 101, 103, 102, 104, 102, 100, 98, 96, 95, 97, 94, 99],
    })

    res = df.select(pl.col.p.qt.compose_by(3))['p']
    res2 = df.select(pq.compose_by(pl.col.p, 3))['p']
    expect = pl.Series([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4], dtype=pl.Int32)
    assert_series_equal(res, expect, check_names=False)
    assert_series_equal(res2, expect, check_names=False)


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

def test_fdiff():
    df = pl.DataFrame({'a': [7, 4, 2, 5, 1, 2]})
    df = df.with_columns(pl.col.a.qt.fdiff(0.5, 4))
    assert_series_equal(df['a'], pl.Series([None, 0.5, -0.875, 3.0625, -2., 0.75]), check_names=False)


def test_linspace():
    expect = np.linspace(-2, 19, 34)
    arr = pq.linspace(-2, 19, 34, eager=True)
    assert_allclose(expect, arr.to_numpy())

def test_cut():
    import pandas as pd
    arr = [1, 3, 5, 1, 5, 6, 7, 32, 1]
    bins = [2, 5, 8]
    labels = [1, 2, 3, 4]
    res = pq.cut(arr, bins, labels, add_bounds=True, eager=True)
    assert_allclose(res.to_numpy(), np.array([1, 2, 2, 1, 2, 3, 3, 4, 1]))
    res = pq.cut(arr, [-100, 2, 5, 8, 100], labels, add_bounds=False, eager=True)

    arr = np.random.rand(100)
    bins = [-0.1, 0.1, 0.3, 0.5, 0.6, 0.8, 1.1]
    labels = [1, 2, 3, 4, 5, 6]
    res = pq.cut(arr, bins, labels, right=True, add_bounds=False, eager=True)
    expect = pd.cut(arr, bins, labels=labels, right=True)
    assert_allclose(res.to_numpy(), expect)

    res = pq.cut(arr, bins, labels, right=False, add_bounds=False, eager=True)
    expect = pd.cut(arr, bins, labels=labels, right=False)
    assert_allclose(res.to_numpy(), expect)

def test_trades():
    from datetime import date
    df = pl.DataFrame({
        'price': [100, 101, 102, 103, 104, 105],
        "signal": [0., 0., 1., 1., -1, -1.],
        "time": pl.date_range(date(2020, 1, 1), date(2020, 1, 6), interval='1d', eager=True),
    })
    trades = df.select(trade=pl.col.signal.qt.to_trades(time='time', price='price'))
    assert_series_equal(
        trades['trade'].struct['time'],
        pl.Series([date(2020, 1, 3), date(2020, 1, 5)], dtype=pl.Datetime("ns")),
        check_names=False
    )
