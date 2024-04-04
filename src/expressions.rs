#![allow(clippy::unused_unit)]
use polars::prelude::*;
use serde::Deserialize;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use std::cmp::min;

#[derive(Deserialize)]
struct TsRankKwargs {
    window: usize,
    min_periods: usize,
    pct: bool,
    rev: bool,
}


#[polars_expr(output_type=Float64)]
fn rolling_rank(inputs: &[Series], kwargs: TsRankKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::Int32 => Ok(impl_ts_rank(s.i32().unwrap(), kwargs).into_series()),
        DataType::Int64 => Ok(impl_ts_rank(s.i64().unwrap(), kwargs).into_series()),
        DataType::Float32 => Ok(impl_ts_rank(s.f32().unwrap(), kwargs).into_series()),
        DataType::Float64 => Ok(impl_ts_rank(s.f64().unwrap(), kwargs).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for rolling_rank, expected Int32, Int64, Float32, Float64."))
        }
    }
}

fn impl_ts_rank<T>(
    ca: &ChunkedArray<T>, 
    kwargs: TsRankKwargs,
) -> Float64Chunked
where
    T: PolarsNumericType,
{   
    let len = ca.len();
    let window = min(len, kwargs.window);
    let min_periods = if window < kwargs.min_periods {
        window
    } else {
        kwargs.min_periods
    };
    let (pct, rev) = (kwargs.pct, kwargs.rev);
    let w_m1 = window - 1;  // window minus one
    let start_iter = std::iter::repeat(0_usize).take(window).chain(1..len-window+1);
    let mut n = 0usize;  // keep the num of valid elements 
    let out: Float64Chunked = ca.into_iter().zip(start_iter).enumerate().map(|(end, (v, start)): (usize, (Option<T::Native>, usize))| {
        let mut n_repeat = 1; // 当前值的重复次数
        let mut rank = 1.; // 先假设为第一名，每当有元素比他更小，排名就加1
        if let Some(v) = v {
            n += 1;
            for i in start..end {
                let a = unsafe{ca.get_unchecked(i)};
                if let Some(a) = a {
                    if a < v {
                        rank += 1.
                    } else if a == v {
                        n_repeat += 1
                    }
                }
            }
        } else {
            rank = f64::NAN
        };
        let out: f64;
        if n >= min_periods {
            let res = if !rev {
                rank + 0.5 * (n_repeat - 1) as f64 // 对于重复值的method: average
            } else {
                (n + 1) as f64 - rank - 0.5 * (n_repeat - 1) as f64
            };
            if pct {
                out = res / n as f64;
            } else {
                out = res;
            }
        } else {
            out = f64::NAN;
        };
        if end >= w_m1 {
            if unsafe{ca.get_unchecked(start)}.is_some() {
                n -= 1;
            };
        }
        if out != out {
            None
        } else {
            Some(out)
        }
    }).collect_trusted();
    out.with_name(ca.name())
}
