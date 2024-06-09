use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;
// use crate::output_func::same_output_type;

#[derive(Deserialize)]
struct CutKwargs {
    right: Option<bool>,
    add_bounds: Option<bool>,
}

#[polars_expr(output_type=Float64)]
pub fn cut(inputs: &[Series], kwargs: CutKwargs) -> PolarsResult<Series> {
    use PlDataType::*;
    let (fac, bin, labels) = (&inputs[0], &inputs[1], &inputs[2]);
    let name = fac.name();
    let right = kwargs.right.unwrap_or(true);
    let add_bounds = kwargs.add_bounds.unwrap_or(true);
    let labels_f64 = labels.cast(&Float64)?;
    let labels = labels_f64.f64()?;
    let res: Float64Chunked = match fac.dtype() {
        PlDataType::Int32 => fac
            .i32()?
            .titer()
            .vcut(bin.cast(&Int32)?.i32()?, labels, right, add_bounds)?
            .try_collect_trusted_vec1()?,
        PlDataType::Int64 => fac
            .i64()?
            .titer()
            .vcut(bin.cast(&Int64)?.i64()?, labels, right, add_bounds)?
            .try_collect_trusted_vec1()?,
        PlDataType::Float32 => fac
            .f32()?
            .titer()
            .vcut(bin.cast(&Float32)?.f32()?, labels, right, add_bounds)?
            .try_collect_trusted_vec1()?,
        PlDataType::Float64 => fac
            .f64()?
            .titer()
            .vcut(bin.cast(&Float64)?.f64()?, labels, right, add_bounds)?
            .try_collect_trusted_vec1()?,
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for cut, expected Int32, Int64, Float32, Float64."))
        }
    };
    Ok(res.with_name(name).into_series())
}
