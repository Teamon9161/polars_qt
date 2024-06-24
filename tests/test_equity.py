import datetime

import polars as pl
from polars.testing import assert_series_equal

from polars_qt import calc_future_ret


def test_calc_ret_single():
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2020, 1, 7)
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(start, end, eager=True),
            "open": [98, 100, 103, 105, 96, 220, 226],
            "close": [100, 102, 105, 96, 90, 226, 220],
            "contract_chg_signal": [0, 0, 0, 0, 0, 1, 0],
            "pos": [0, 1, 1, 0.5, -0.5, -0.5, 0],
            "spread": [0] * 7,
        }
    )
    config = {
        "open": "open",
        "close": "close",
        "is_signal": False,
        "contract_chg_signal": "contract_chg_signal",
        "init_cash": 1_000_000,
        "c_rate": 3e-4,
        "multiplier": 10,
        "leverage": 2,
    }
    out = df.select(
        equity1=calc_future_ret(signal="pos", **config),
        equity2=pl.col("pos").qt.calc_future_ret(slippage="spread", **config),
    )

    expect = pl.Series(
        [
            1_000_000,
            1_039_400,
            1_099_400,
            1_004_869.805,
            1067027.021,
            1037286.821,
            1036957.991,
        ],
    )
    assert_series_equal(out["equity1"], expect, check_names=False)
    assert_series_equal(out["equity2"], expect, check_names=False)

def test_calc_tick_ret():
    df = pl.DataFrame(
        {
            "bid": [101, 102, 103, 104, 103, 101, 206, 204, 208, 204, 202, 201],
            "ask": [102, 103, 104, 105, 104, 102, 207, 205, 209, 205, 203, 202],
            "contract_chg_signal": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "signal": [0, 1, 1, 1, 1, 1, 1, 0, -1, -1, 1, 1],
        }
    )
    kwargs = {
        "is_signal": True,
        "contract_chg_signal": "contract_chg_signal",
        "init_cash": 10_000,
        "c_rate": 1e-4,
        "multiplier": 1,
    }
    expect = pl.Series("cash", [
        10000., 10000., 10047.5009, 10144.5009, 10047.5009, 9853.5009, 9754.5318, 9660.5318,
        9636.073, 9796.1162, 9888.1162, 9791.208,
    ])
    df = df.with_columns(cash = pl.col.signal.qt.calc_tick_future_ret(pl.col.bid, pl.col.ask, **kwargs))
    assert_series_equal(df['cash'], expect)
