use super::from_input::FromInput;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use tea_strategy::{AutoBollKwargs, BollKwargs, StrategyFilter};

#[polars_expr(output_type=Float64)]
fn boll(inputs: &[Series], kwargs: BollKwargs) -> PolarsResult<Series> {
    let fac = inputs[0].f64()?;
    let filter = if inputs.len() == 5 {
        Some(StrategyFilter::from_inputs(inputs, &[1, 2, 3, 4])?)
    } else if inputs.len() == 1 {
        None
    } else {
        polars_bail!(ComputeError: "wrong lenght of inputs in function boll")
    };
    let out: Float64Chunked = tea_strategy::boll(fac, filter, &kwargs);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn auto_boll(inputs: &[Series], kwargs: AutoBollKwargs) -> PolarsResult<Series> {
    let fac = inputs[0].f64()?;
    let filter = if inputs.len() == 5 {
        Some(StrategyFilter::from_inputs(inputs, &[1, 2, 3, 4])?)
    } else if inputs.len() == 1 {
        None
    } else {
        polars_bail!(ComputeError: "wrong lenght of inputs in function auto boll")
    };
    let out: Float64Chunked = tea_strategy::auto_boll(fac, filter, &kwargs);
    Ok(out.into_series())
}
