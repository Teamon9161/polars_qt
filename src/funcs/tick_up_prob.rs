use gauss_quad::Midpoint;
use polars::prelude::arity::binary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::f64::consts::PI;
use tea_strategy::tevec::prelude::Cast;

#[inline]
fn up_prob(n_ask: f64, n_bid: f64, quad_method: &Midpoint) -> f64 {
    let f = |t: f64| -> f64 {
        (2. - t.cos() - ((2. - t.cos()) * (2. - t.cos()) - 1.).sqrt()).powi(n_ask as i32)
            * (n_bid * t).sin()
            * (t * 0.5).cos()
            / (t * 0.5).sin()
    };
    1. / PI * quad_method.integrate(0.0, PI, f)
}

#[inline]
pub fn tick_up_prob_rs<T1, T2>(
    n_ask: &ChunkedArray<T1>,
    n_bid: &ChunkedArray<T2>,
    degree: Option<usize>,
) -> Float64Chunked
where
    T1: PolarsDataType + PolarsNumericType,
    T1::Native: Cast<f64>,
    T2: PolarsDataType + PolarsNumericType,
    T2::Native: Cast<f64>,
{
    let quad_method = Midpoint::new(degree.unwrap_or(1_000_000)).unwrap();
    binary_elementwise(n_ask, n_bid, |n_ask, n_bid| match (n_ask, n_bid) {
        (Some(n_ask), Some(n_bid)) => Some(up_prob(n_ask.cast(), n_bid.cast(), &quad_method)),
        _ => None,
    })
}

#[derive(Deserialize)]
pub struct TickUpProbKwargs {
    degree: Option<usize>,
}

#[polars_expr(output_type=Float64)]
fn tick_up_prob(inputs: &[Series], kwargs: TickUpProbKwargs) -> PolarsResult<Series> {
    let n_ask = &inputs[0];
    let n_bid = &inputs[1];
    let name = n_ask.name();
    use DataType::*;
    let out = match (n_ask.dtype(), n_bid.dtype()) {
        (Int32, Int32) => tick_up_prob_rs(n_ask.i32()?, n_bid.i32()?, kwargs.degree),
        (Int64, Int64) => tick_up_prob_rs(n_ask.i64()?, n_bid.i64()?, kwargs.degree),
        (Float32, Float32) => tick_up_prob_rs(n_ask.f32()?, n_bid.f32()?, kwargs.degree),
        (Float64, Float64) => tick_up_prob_rs(n_ask.f64()?, n_bid.f64()?, kwargs.degree),
        (Int32, Int64) => tick_up_prob_rs(n_ask.i32()?, n_bid.i64()?, kwargs.degree),
        (Int64, Int32) => tick_up_prob_rs(n_ask.i64()?, n_bid.i32()?, kwargs.degree),
        (Float32, Float64) => tick_up_prob_rs(n_ask.f32()?, n_bid.f64()?, kwargs.degree),
        (Float64, Float32) => tick_up_prob_rs(n_ask.f64()?, n_bid.f32()?, kwargs.degree),
        (dtype1, dtype2) => {
            polars_bail!(InvalidOperation: "dtype1 {dtype1} and dtype2 {dtype2} not
            supported for tick_up_prob, expected Int32, Int64, Float32, Float64.")
        }
    };
    Ok(out.with_name(name.clone()).into_series())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_up_prob() {
        let quad_method = Midpoint::new(1_000_000).unwrap();
        let res = up_prob(2., 1., &quad_method);
        assert!((res - 0.30234727368).abs() < 1e-10)
    }
}
