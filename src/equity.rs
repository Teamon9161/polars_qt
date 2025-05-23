use crate::auto_cast;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use tea_strategy::equity;
use tea_strategy::equity::{
    profit_vec_to_series, FutureRetKwargs, FutureRetSpreadKwargs, TickFutureRetFullKwargs,
    TickFutureRetKwargs,
};

#[polars_expr(output_type=Float64)]
fn calc_future_ret(inputs: &[Series], kwargs: FutureRetKwargs) -> PolarsResult<Series> {
    let (pos, opening_cost, closing_cost) = (&inputs[0], &inputs[1], &inputs[2]);
    let (pos, opening_cost, closing_cost) = auto_cast!(Float64(pos, opening_cost, closing_cost));
    let contract_chg_signal = if inputs.len() == 3 {
        None
    } else {
        Some(auto_cast!(Boolean(inputs[3])))
    };
    let out: Float64Chunked = equity::calc_future_ret(
        pos.f64()?,
        opening_cost.f64()?,
        closing_cost.f64()?,
        contract_chg_signal.as_ref().map(|s| s.bool().unwrap()),
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
    let (pos, opening_cost, closing_cost, spread) =
        auto_cast!(Float64(pos, opening_cost, closing_cost, spread));
    let contract_chg_signal = if inputs.len() == 4 {
        None
    } else {
        Some(auto_cast!(Boolean(inputs[4])))
    };
    let out: Float64Chunked = equity::calc_future_ret_with_spread(
        pos.f64()?,
        opening_cost.f64()?,
        closing_cost.f64()?,
        spread.f64()?,
        contract_chg_signal.as_ref().map(|s| s.bool().unwrap()),
        &kwargs,
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn calc_tick_future_ret(inputs: &[Series], kwargs: TickFutureRetKwargs) -> PolarsResult<Series> {
    let (signal, bid, ask) = (&inputs[0], &inputs[1], &inputs[2]);
    let (signal, bid, ask) = auto_cast!(Float64(signal, bid, ask));
    let contract_chg_signal = if inputs.len() == 3 {
        None
    } else {
        Some(auto_cast!(Boolean(inputs[3])))
    };
    let out: Float64Chunked = equity::calc_tick_future_ret(
        signal.f64()?,
        bid.f64()?,
        ask.f64()?,
        contract_chg_signal.as_ref().map(|s| s.bool().unwrap()),
        &kwargs,
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn calc_tick_future_ret_full(
    inputs: &[Series],
    kwargs: TickFutureRetFullKwargs,
) -> PolarsResult<Series> {
    let (signal, bid, ask) = (&inputs[0], &inputs[1], &inputs[2]);
    let (signal, bid, ask) = auto_cast!(Float64(signal, bid, ask));
    let contract_chg_signal = if inputs.len() == 3 {
        None
    } else {
        Some(auto_cast!(Boolean(inputs[3])))
    };
    let profit_vec = equity::calc_tick_future_ret_full(
        signal.f64()?,
        bid.f64()?,
        ask.f64()?,
        contract_chg_signal.as_ref().map(|s| s.bool().unwrap()),
        &kwargs,
    );
    let out = profit_vec_to_series(&profit_vec);
    Ok(out)
}
