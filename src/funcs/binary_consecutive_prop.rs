use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use tea_strategy::tevec::prelude::*;

#[derive(Deserialize)]
struct BinaryConsecutivePropKwargs {
    window: usize,
    min_periods: Option<usize>,
}

fn impl_binary_consecutive_prop(
    arr: &BooleanChunked,
    window: usize,
    min_periods: Option<usize>,
) -> Float64Chunked {
    let min_periods = min_periods.unwrap_or(window / 2).min(window);
    let mut n = 0;
    let mut true_count = 0;
    let mut false_count = 0;
    let out = arr.rolling_apply(
        window,
        |v_rm, v| {
            if let Some(v) = v {
                n += 1;
                if v {
                    true_count += 1;
                } else {
                    false_count += 1;
                }
            }
            let res = if n >= min_periods {
                if n == 1 {
                    Some(0.5)
                } else if let Some(v) = v {
                    if v {
                        Some((true_count - 1) as f64 / (n - 1) as f64)
                    } else {
                        Some((false_count - 1) as f64 / (n - 1) as f64)
                    }
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(Some(v_rm)) = v_rm {
                n -= 1;
                if v_rm {
                    true_count -= 1;
                } else {
                    false_count -= 1;
                }
            }
            res
        },
        None,
    );
    out.unwrap()
}

#[polars_expr(output_type=Float64)]
pub fn binary_consecutive_prop(
    inputs: &[Series],
    kwargs: BinaryConsecutivePropKwargs,
) -> PolarsResult<Series> {
    let s = &inputs[0];
    let name = s.name();
    let s = crate::auto_cast!(Boolean(s));
    let res = impl_binary_consecutive_prop(s.bool()?, kwargs.window, kwargs.min_periods);
    Ok(res.into_series().with_name(name.clone()))
}
