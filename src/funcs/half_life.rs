use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;

#[derive(Deserialize)]
struct HalfLifeKwargs {
    min_periods: Option<usize>,
}

#[polars_expr(output_type=Int32)]
pub fn half_life(inputs: &[Series], kwargs: HalfLifeKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let res = match s.dtype() {
        PlDataType::Int32 => s.i32()?.half_life(kwargs.min_periods),
        PlDataType::Int64 => s.i64()?.half_life(kwargs.min_periods),
        PlDataType::Float32 => s.f32()?.half_life(kwargs.min_periods),
        PlDataType::Float64 => s.f64()?.half_life(kwargs.min_periods),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for half_life, expected Int32, Int64, Float32, Float64."))
        }
    };
    Ok(Series::new(name, vec![res as i32]))
}
