use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;

#[derive(Deserialize)]
struct TsRankKwargs {
    window: usize,
    min_periods: Option<usize>,
    pct: bool,
    rev: bool,
}

#[polars_expr(output_type=Float64)]
fn rolling_rank(inputs: &[Series], kwargs: TsRankKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let out: Float64Chunked = match s.dtype() {
        PlDataType::Int32 => {
            s.i32()?
                .ts_vrank(kwargs.window, kwargs.min_periods, kwargs.pct, kwargs.rev)
        }
        PlDataType::Int64 => {
            s.i64()?
                .ts_vrank(kwargs.window, kwargs.min_periods, kwargs.pct, kwargs.rev)
        }
        PlDataType::Float32 => {
            s.f32()?
                .ts_vrank(kwargs.window, kwargs.min_periods, kwargs.pct, kwargs.rev)
        }
        PlDataType::Float64 => {
            s.f64()?
                .ts_vrank(kwargs.window, kwargs.min_periods, kwargs.pct, kwargs.rev)
        }
        dtype => {
            polars_bail!(InvalidOperation: "dtype {dtype} not \
            supported for rolling_rank, expected Int32, Int64, Float32, Float64.")
        }
    };
    Ok(out.with_name(name).into_series())
}
