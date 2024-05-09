use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
// use serde::Deserialize;

use tea_strategy::equity;
use tea_strategy::equity::{FutureRetKwargs, FutureRetSpreadKwargs};

#[polars_expr(output_type=Float64)]
fn calc_future_ret(inputs: &[Series], kwargs: FutureRetKwargs) -> PolarsResult<Series> {
    let (pos, opening_cost, closing_cost) = (&inputs[0], &inputs[1], &inputs[2]);
    let pos = pos.f64()?;
    let opening_cost = opening_cost.f64()?;
    let closing_cost = closing_cost.f64()?;
    let contract_chg_signal = if inputs.len() == 3 {
        None
    } else {
        Some(inputs[3].bool()?)
    };
    let out: Float64Chunked = equity::calc_future_ret(
        pos,
        opening_cost,
        closing_cost,
        contract_chg_signal,
        &kwargs,
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn calc_future_ret_with_spread(
    inputs: &[Series],
    kwargs: FutureRetSpreadKwargs,
) -> PolarsResult<Series> {
    let (pos, opening_cost, closing_cost, spread) =
        (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    let pos = pos.f64()?;
    let opening_cost = opening_cost.f64()?;
    let closing_cost = closing_cost.f64()?;
    let spread = spread.f64()?;
    let contract_chg_signal = if inputs.len() == 4 {
        None
    } else {
        Some(inputs[4].bool()?)
    };
    let out: Float64Chunked = equity::calc_future_ret_with_spread(
        pos,
        opening_cost,
        closing_cost,
        spread,
        contract_chg_signal,
        &kwargs,
    );
    Ok(out.into_series())
}
