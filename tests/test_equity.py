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
