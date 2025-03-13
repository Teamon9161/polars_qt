use polars::prelude::arity::unary_elementwise;
use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;

#[derive(Deserialize)]
struct TsZscoreKwargs {
    window: usize,
    min_periods: Option<usize>,
}

#[polars_expr(output_type=Float64)]
fn rolling_zscore(inputs: &[Series], kwargs: TsZscoreKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let out: Float64Chunked = match s.dtype() {
        PlDataType::Int32 => s.i32()?.ts_vzscore(kwargs.window, kwargs.min_periods),
        PlDataType::Int64 => s.i64()?.ts_vzscore(kwargs.window, kwargs.min_periods),
        PlDataType::Float32 => s.f32()?.ts_vzscore(kwargs.window, kwargs.min_periods),
        PlDataType::Float64 => s.f64()?.ts_vzscore(kwargs.window, kwargs.min_periods),
        dtype => {
            polars_bail!(InvalidOperation: "dtype {dtype} not \
            supported for rolling_zscore, expected Int32, Int64, Float32, Float64.")
        }
    };
    Ok(out.with_name(name.clone()).into_series())
}

#[derive(Deserialize)]
struct ZscoreKwargs {
    min_periods: Option<usize>,
}

#[allow(clippy::unnecessary_unwrap)]
fn calc_zscore<T>(s: &ChunkedArray<T>, min_periods: Option<usize>) -> Float64Chunked
where
    T: PolarsNumericType,
    for<'a> &'a ChunkedArray<T>: AggValidBasic<Option<T::Native>>,
    T::Native: IsNone<Inner = T::Native> + Number,
{
    use tevec::prelude::EPS;
    let mean = s.vmean();
    let std = s.vstd(min_periods.unwrap_or(2));
    unary_elementwise(s, |v| {
        if v.is_none() || std.is_none() || std.abs() < EPS {
            None
        } else {
            Some((v.unwrap().f64() - mean) / std)
        }
    })
}

#[polars_expr(output_type=Float64)]
fn zscore(inputs: &[Series], kwargs: ZscoreKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let out: Float64Chunked = match s.dtype() {
        PlDataType::Int32 => calc_zscore(s.i32()?, kwargs.min_periods),
        PlDataType::Int64 => calc_zscore(s.i64()?, kwargs.min_periods),
        PlDataType::Float32 => calc_zscore(s.f32()?, kwargs.min_periods),
        PlDataType::Float64 => calc_zscore(s.f64()?, kwargs.min_periods),
        dtype => {
            polars_bail!(InvalidOperation: "dtype {dtype} not \
            supported for zscore, expected Int32, Int64, Float32, Float64.")
        }
    };
    Ok(out.with_name(name.clone()).into_series())
}
