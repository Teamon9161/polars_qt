#[macro_export]
macro_rules! define_strategy {
    ($strategy: ident $({$mark: tt})?, $kwargs: ty) => {
        #[polars_expr(output_type=Float64)]
        fn $strategy(inputs: &[Series], kwargs: $kwargs) -> PolarsResult<Series> {
            let filter = if inputs.len() == 5 {
                Some(StrategyFilter::from_inputs(inputs, &[1, 2, 3, 4])?)
            } else if inputs.len() == 1 {
                None
            } else {
                polars_bail!(ComputeError: format!("wrong length of inputs in function {}", stringify!($strategy)))

            };
            let fac = &inputs[0];
            let out: Float64Chunked = match fac.dtype() {
                DataType::Int32 => tea_strategy::$strategy(fac.i32()?, filter.as_ref(), &kwargs)$($mark)?,
                DataType::Int64 => tea_strategy::$strategy(fac.i64()?, filter.as_ref(), &kwargs)$($mark)?,
                DataType::Float32 => tea_strategy::$strategy(fac.f32()?, filter.as_ref(), &kwargs)$($mark)?,
                DataType::Float64 => tea_strategy::$strategy(fac.f64()?, filter.as_ref(), &kwargs)$($mark)?,
                dtype => polars_bail!(InvalidOperation: format!("dtype {} not supported for {}", dtype, stringify!($strategy))),
            };
            Ok(out.into_series())
        }
    };
}
