use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::export::num::ToPrimitive};

#[polars_expr(output_type=Int32)]
fn compose_by(inputs: &[Series]) -> PolarsResult<Series> {
    let (expr, by) = (&inputs[0], &inputs[1]);
    polars_ensure!(by.len() == 1, ComputeError: "By should be a scalar value.");
    let value: f64 = match by.dtype() {
        DataType::Int32 => by.i32()?.get(0).unwrap() as f64,
        DataType::Int64 => by.i64()?.get(0).unwrap() as f64,
        DataType::Float32 => by.f32()?.get(0).unwrap() as f64,
        DataType::Float64 => by.f64()?.get(0).unwrap(),
        dtype => polars_bail!(InvalidOperation:format!("dtype of value: {dtype} not \
        supported for compose_by, expected Int32, Int64, Float32, Float64.")),
    };
    match expr.dtype() {
        DataType::Int32 => Ok(impl_compose_by(expr.i32().unwrap(), value).into_series()),
        DataType::Int64 => Ok(impl_compose_by(expr.i64().unwrap(), value).into_series()),
        DataType::Float32 => Ok(impl_compose_by(expr.f32().unwrap(), value).into_series()),
        DataType::Float64 => Ok(impl_compose_by(expr.f64().unwrap(), value).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype of expr {dtype} not \
            supported for compose_by, expected Int32, Int64, Float32, Float64."))
        }
    }
}

fn impl_compose_by<T>(arr: &ChunkedArray<T>, value: f64) -> ChunkedArray<Int32Type>
where
    T: PolarsNumericType,
{
    let mut acc = 0.;
    let mut group = 0;
    arr.apply_generic(|x| {
        if x.is_none() {
            return Some(group);
        }
        let x = x.unwrap();
        acc += x.to_f64().unwrap();
        if acc.abs() >= value {
            group += 1;
            acc = 0.;
        }
        Some(group)
    })
}
