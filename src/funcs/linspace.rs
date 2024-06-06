use polars::prelude::DataType as PlDataType;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use tea_strategy::tevec::prelude::*;

#[polars_expr(output_type=Float64)]
pub fn linspace(inputs: &[Series]) -> PolarsResult<Series> {
    let (start, end, num) = (&inputs[0], &inputs[1], &inputs[2]);
    let name = start.name();
    polars_ensure!(
        (start.len() == 1) && (end.len() == 1) && (num.len() == 1),
        ComputeError: "linspace expects all inputs to be scalars"
    );
    use PlDataType::*;
    let arr: Float64Chunked = Vec1Create::linspace(
        Some(start.cast(&Float64)?.f64()?.get(0).unwrap()),
        end.cast(&Float64)?.f64()?.get(0).unwrap(),
        num.cast(&Int32)?.i32()?.get(0).unwrap() as usize,
    );
    Ok(arr.with_name(name).into_series())
}
