import polars as pl
from polars.testing import assert_series_equal

import polars_qt


def test_boll():
    df = pl.DataFrame({
        'close': [10., 11, 12, 10, 11, 12, 10, 11, 12, 13, 14, 10, 7, 5, 4, 3, 4, 4, 3, 2],
        'short_open_filter': [1] * 11 + [0, 0, 1, 1, 1] + [1] * 4,
    })
    df = df.with_columns([
        pl.col('close').qt.boll((4, 1), delay_open=False).alias('s1'), # base boll
        # boll with filters
        pl.col('close').qt.boll((4, 1), filters=[
            pl.col('close') > 0, False,  # long open, long stop
            'short_open_filter', False,  # short open, short stop
        ], delay_open=False).alias('s2'),
        # boll with filters and delay open
        pl.col('close').qt.boll((4, 1), filters=[
            pl.col('close') > 0, False,  # long open, long stop
            'short_open_filter', False,  # short open, short stop
        ], delay_open=True).alias('s3'),
        # boll with fac_vol stop, take profit if close reaches 5 * fac_vol
        pl.col('close').qt.boll((4, 1, 5), fac_vol=pl.repeat(1, pl.col('close').len())).alias('s4'),
    ])
    expect1 = pl.Series('s1', [0., 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, -1, 0, 0, 0, -1])
    assert_series_equal(df['s1'], expect1)
    expect2 = pl.Series('s2', [0., 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1])
    assert_series_equal(df['s2'], expect2)
    expect3 = pl.Series('s3', [0., 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0, -1])
    assert_series_equal(df['s3'], expect3)
    expect3 = pl.Series('s4', [0., 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, -1, 0, 0, 0, -1])
    assert_series_equal(df['s4'], expect3)
test_boll()
