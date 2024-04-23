import polars as pl
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
