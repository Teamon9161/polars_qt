#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[1];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn if_then(inputs: &[Series]) -> PolarsResult<Series> {
    let cond = inputs[0].bool()?;
    if cond.len() != 1 {
        return Err(PolarsError::InvalidOperation(
            "if_then expects a single boolean value".into(),
        ));
    };
    let cond = cond.get(0).unwrap();
    if cond {
        Ok(inputs[1].clone())
    } else {
        Ok(inputs[2].clone())
    }
}
