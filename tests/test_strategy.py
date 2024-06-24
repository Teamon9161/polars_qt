import polars as pl
from polars.testing import assert_series_equal

import polars_qt


def test_boll():
    df = pl.DataFrame({
        'close': [10., 11, 11.9, 10, 11, 12, 10, 11, 12, 13, 14, 10, 7, 5, 4, 3, 4, 4, 3, 2],
        'short_open_filter': [1] * 11 + [0, 0, 1, 1, 1] + [1] * 4,
    })
    df = df.with_columns([
        pl.col('close').rolling_mean(4, min_periods=2).alias('mean'),
        pl.col('close').rolling_std(4, min_periods=2).alias('std'),
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
    ])#.with_columns(upper=pl.col('mean')+pl.col('std'), down=pl.col('mean')-pl.col('std')).to_pandas()

    expect1 = pl.Series('s1', [0., 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, -1, 0, 0, 0, -1])
    assert_series_equal(df['s1'], expect1)
    expect2 = pl.Series('s2', [0., 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1])
    assert_series_equal(df['s2'], expect2)
    expect3 = pl.Series('s3', [0., 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0, -1])
    assert_series_equal(df['s3'], expect3)

def test_prob_thres():
    df = pl.DataFrame({
        'prob': [0.3, 0.6, 0.7, 0.6, 0.4, 0.2, 0.5, 0.4],
    })
    kwargs = {
        'thresholds': (0.6, 0.5, 0.4, 0.5),
        'per_hand': 1.,
        'max_hand': 2.,
    }
    df = df.with_columns(res=pl.col.prob.qt.prob_threshold(**kwargs))
    expect = pl.Series('res', [-1., 1., 2., 2., -1., -2., 0., -1.])
    assert_series_equal(df['res'], expect)
