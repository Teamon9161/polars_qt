use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tevec::rolling::*;

#[derive(Deserialize)]
struct FdiffKwargs {
    d: f64,
    window: usize,
    ignore_na: Option<bool>,
    min_periods: Option<usize>,
}

#[polars_expr(output_type=Float64)]
fn fdiff(inputs: &[Series], kwargs: FdiffKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let ignore_na = kwargs.ignore_na.unwrap_or(true);
    let out: Float64Chunked = if !ignore_na {
        match s.dtype() {
            PlDataType::Int32 => s.i32()?.ts_fdiff(kwargs.d, kwargs.window),
            PlDataType::Int64 => s.i64()?.ts_fdiff(kwargs.d, kwargs.window),
            PlDataType::Float32 => s.f32()?.ts_fdiff(kwargs.d, kwargs.window),
            PlDataType::Float64 => s.f64()?.ts_fdiff(kwargs.d, kwargs.window),
            dtype => {
                polars_bail!(InvalidOperation: "dtype {dtype} not \
                supported for fdiff, expected Int32, Int64, Float32, Float64.")
            }
        }
    } else {
        match s.dtype() {
            PlDataType::Int32 => s
                .i32()?
                .ts_vfdiff(kwargs.d, kwargs.window, kwargs.min_periods),
            PlDataType::Int64 => s
                .i64()?
                .ts_vfdiff(kwargs.d, kwargs.window, kwargs.min_periods),
            PlDataType::Float32 => s
                .f32()?
                .ts_vfdiff(kwargs.d, kwargs.window, kwargs.min_periods),
            PlDataType::Float64 => s
                .f64()?
                .ts_vfdiff(kwargs.d, kwargs.window, kwargs.min_periods),
            dtype => {
                polars_bail!(InvalidOperation: "dtype {dtype} not \
                supported for fdiff, expected Int32, Int64, Float32, Float64.")
            }
        }
    };
    Ok(out.with_name(name.clone()).into_series())
}
