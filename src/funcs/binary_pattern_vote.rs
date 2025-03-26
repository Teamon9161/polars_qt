use polars::prelude::arity::binary_elementwise_for_each;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;

#[derive(Deserialize)]
struct BinaryPatternVoteKwargs {
    lookup_len: usize,
    pattern_len: usize,
    alpha: f64,
    lambda: f64,
    predict_n: Option<usize>,
}

fn binary_distance(a1: &BooleanChunked, a2: &BooleanChunked, alpha: f64) -> f64 {
    let mut i = 0;
    let mut dist = 0f64;
    debug_assert!(a1.len() == a2.len());
    let n = a1.len();
    binary_elementwise_for_each(a1, a2, |v1, v2| {
        match (v1, v2) {
            (Some(v1), Some(v2)) => {
                if v1 != v2 {
                    dist += alpha.powi((n - i) as i32);
                }
            }
            (Some(_v1), None) => {
                dist += alpha.powi((n - i) as i32);
            }
            (None, Some(_v2)) => {
                dist += alpha.powi((n - i) as i32);
            }
            (None, None) => {}
        }
        i += 1;
    });
    dist
}

fn impl_binary_pattern_vote(
    arr: &BooleanChunked,
    lookup_len: usize,
    pattern_len: usize,
    alpha: f64,
    lambda: f64,
    predict_n: Option<usize>,
) -> PolarsResult<ChunkedArray<Float64Type>> {
    let predict_n = predict_n.unwrap_or(1);
    if lookup_len < pattern_len + predict_n {
        polars_bail!(InvalidOperation:format!("lookup length: {lookup_len} must be greater than pattern length + predict_n: {}", pattern_len + predict_n))
    }
    use std::f64::consts::E;
    let out = arr
        .rolling_custom::<Float64Chunked, _, _>(
            lookup_len,
            |data| {
                if data.len() < pattern_len + predict_n {
                    return None;
                }
                let current_pattern = data.slice(-(pattern_len as i64), pattern_len);
                debug_assert!(current_pattern.len() == pattern_len);
                let mut up_sum = 0.;
                let mut all_sum = 0.;
                for i in 0..(data.len() - pattern_len - predict_n + 1) {
                    let past_pattern = data.slice(i as i64, pattern_len);
                    let past_predict = if predict_n == i {
                        data.get(i + pattern_len)
                            .map(|v| v as i8 as f64)
                            .unwrap_or(0.5)
                    } else {
                        data.slice((i + pattern_len) as i64, predict_n)
                            .mean()
                            .unwrap_or(0.5)
                    };

                    let dist = binary_distance(&past_pattern, &current_pattern, alpha);
                    let dist = E.powf(-lambda * dist);
                    up_sum += dist * past_predict;
                    all_sum += dist;
                }
                if all_sum == 0. {
                    None
                } else {
                    Some(up_sum / all_sum)
                }
            },
            None,
        )
        .unwrap();
    Ok(out)
}

#[polars_expr(output_type=Float64)]
pub fn binary_pattern_vote(
    inputs: &[Series],
    kwargs: BinaryPatternVoteKwargs,
) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let s = crate::auto_cast!(Boolean(s));
    let res = impl_binary_pattern_vote(
        s.bool()?,
        kwargs.lookup_len,
        kwargs.pattern_len,
        kwargs.alpha,
        kwargs.lambda,
        kwargs.predict_n,
    )?;
    Ok(res.into_series().with_name(name.clone()))
}
