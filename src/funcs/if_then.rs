use crate::output_func::same_output_type;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type_func=same_output_type)]
fn if_then(inputs: &[Series]) -> PolarsResult<Series> {
    let cond = inputs[0].bool()?;
    polars_ensure!(
        cond.len() == 1,
        ComputeError: "if_then expects a single boolean value",
    );
    let cond = cond.get(0).unwrap();
    if cond {
        Ok(inputs[1].clone())
    } else {
        Ok(inputs[2].clone())
    }
}
