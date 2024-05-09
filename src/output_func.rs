use polars::prelude::*;

pub fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[1];
    Ok(field.clone())
}
