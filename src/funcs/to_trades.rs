use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use tea_strategy::{signal_to_trades, trade_vec_to_series};
use tevec::prelude::{unit, DateTime, TIter};

#[polars_expr(output_type=Float64)]
pub fn to_trades(inputs: &[Series]) -> PolarsResult<Series> {
    use PlDataType::*;
    let signal = &inputs[0].cast(&Float64)?;
    let name = signal.name();
    let time = &inputs[1].cast(&Datetime(TimeUnit::Nanoseconds, None))?;
    let trades = match inputs.len() {
        3 => {
            let price = &inputs[2].cast(&Float64)?;
            signal_to_trades(
                signal.f64()?.titer(),
                price.f64()?.titer().into(),
                TIter::<DateTime<unit::Nanosecond>>::titer(&time.datetime()?),
            )
        }
        4 => {
            let bid_price = &inputs[2].cast(&Float64)?;
            let ask_price = &inputs[3].cast(&Float64)?;
            signal_to_trades(
                signal.f64()?.titer(),
                (bid_price.f64()?.titer(), ask_price.f64()?.titer()).into(),
                TIter::<DateTime<unit::Nanosecond>>::titer(&time.datetime()?),
            )
        }
        _ => {
            polars_bail!(ComputeError:
                "invalid number of arguments, arguments must be 3 or 4"
            );
        }
    };
    Ok(trade_vec_to_series(&trades).with_name(name))
}
