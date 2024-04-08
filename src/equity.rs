#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
// use polars::prelude::arity::binary_elementwise;
use serde::Deserialize;

#[derive(Deserialize)]
struct FutureRetKwargs {
    init_cash: usize,
    multiplier: f64,
    leverage: f64,
    slippage: f64,
    ticksize: f64,
    c_rate: f64,
    blowup: bool,
    commision_type: String,
}

#[derive(Deserialize)]
struct FutureRetSpreadKwargs {
    init_cash: usize,
    multiplier: f64,
    leverage: f64,
    c_rate: f64,
    blowup: bool,
    commision_type: String,
}

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
    Ok(
        impl_calc_future_ret(pos, opening_cost, closing_cost, contract_chg_signal, kwargs)
            .into_series(),
    )
}

fn impl_calc_future_ret(
    pos_se: &ChunkedArray<Float64Type>,
    opening_cost_se: &ChunkedArray<Float64Type>,
    closing_cost_se: &ChunkedArray<Float64Type>,
    contract_chg_signal_se: Option<&ChunkedArray<BooleanType>>,
    kwargs: FutureRetKwargs,
) -> Float64Chunked {
    let mut cash = kwargs.init_cash as f64;
    let mut last_pos = 0_f64; // pos_arr[0];
    let mut last_lot_num = 0.;
    if pos_se.is_empty() {
        return Float64Chunked::from_vec(pos_se.name(), vec![]);
    }
    let mut last_close = closing_cost_se.get(0).unwrap();
    let blowup = kwargs.blowup;
    let multiplier = kwargs.multiplier;
    let commision_type = kwargs.commision_type;
    let slippage = kwargs.slippage;
    let ticksize = kwargs.ticksize;
    let leverage = kwargs.leverage;
    let c_rate = kwargs.c_rate;
    if let Some(contract_chg_signal_se) = contract_chg_signal_se {
        pos_se
            .into_iter()
            .zip(opening_cost_se.into_no_null_iter())
            .zip(closing_cost_se.into_no_null_iter())
            .zip(contract_chg_signal_se.into_no_null_iter())
            .map(|(((pos, opening_cost), closing_cost), contract_signal)| {
                if pos.is_none() {
                    return Some(cash);
                } else if blowup && cash <= 0. {
                    return Some(0.);
                }
                let pos = pos.unwrap();
                if (last_lot_num != 0.) && (!contract_signal) {
                    // 换月的时候不计算跳开的损益
                    cash +=
                        last_lot_num * (opening_cost - last_close) * multiplier * last_pos.signum();
                }
                // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                if (pos != last_pos) || contract_signal {
                    // 仓位出现变化，计算新的理论持仓手数
                    let l = ((cash * leverage * pos.abs()) / (multiplier * opening_cost)).floor();
                    let (lot_num, lot_num_change) = if !contract_signal {
                        (
                            l,
                            (l * pos.signum() - last_lot_num * last_pos.signum()).abs(),
                        )
                    } else {
                        (l, l.abs() * 2.)
                    };
                    // 扣除手续费变动
                    if &commision_type == "percent" {
                        cash -= lot_num_change
                            * multiplier
                            * (opening_cost * c_rate + slippage * ticksize);
                    } else {
                        cash -= lot_num_change * (c_rate + multiplier * slippage * ticksize);
                    };
                    // 更新上期持仓手数和持仓头寸
                    last_lot_num = lot_num;
                    last_pos = pos;
                }
                // 计算当期损益
                if last_lot_num != 0. {
                    cash += last_lot_num
                        * last_pos.signum()
                        * (closing_cost - opening_cost)
                        * multiplier;
                }
                last_close = closing_cost; // 更新上期收盘价
                Some(cash)
            })
            .collect_trusted()
    } else {
        // 不考虑合约换月信号的情况
        pos_se
            .into_iter()
            .zip(opening_cost_se.into_no_null_iter())
            .zip(closing_cost_se.into_no_null_iter())
            .map(|((pos, opening_cost), closing_cost)| {
                if pos.is_none() {
                    return Some(cash);
                } else if blowup && cash <= 0. {
                    return Some(0.);
                }
                let pos = pos.unwrap();
                if last_lot_num != 0. {
                    cash +=
                        last_lot_num * (opening_cost - last_close) * multiplier * last_pos.signum();
                }
                // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                if pos != last_pos {
                    // 仓位出现变化
                    // 计算新的理论持仓手数
                    let lot_num =
                        ((cash * leverage * pos.abs()) / (multiplier * opening_cost)).floor();
                    // 扣除手续费变动
                    if &commision_type == "percent" {
                        cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum()).abs()
                            * multiplier
                            * (opening_cost * c_rate + slippage * ticksize);
                    } else {
                        cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum()).abs()
                            * (c_rate + multiplier * slippage * ticksize);
                    };
                    // 更新上期持仓手数和持仓头寸
                    last_lot_num = lot_num;
                    last_pos = pos;
                }
                // 计算当期损益
                if last_lot_num != 0. {
                    cash += last_lot_num
                        * (closing_cost - opening_cost)
                        * multiplier
                        * last_pos.signum();
                }
                last_close = closing_cost; // 更新上期收盘价
                Some(cash)
            })
            .collect_trusted()
    }
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
    Ok(impl_calc_future_ret_with_spread(
        pos,
        opening_cost,
        closing_cost,
        spread,
        contract_chg_signal,
        kwargs,
    )
    .into_series())
}

fn impl_calc_future_ret_with_spread(
    pos_se: &ChunkedArray<Float64Type>,
    opening_cost_se: &ChunkedArray<Float64Type>,
    closing_cost_se: &ChunkedArray<Float64Type>,
    spread_se: &ChunkedArray<Float64Type>,
    contract_chg_signal_se: Option<&ChunkedArray<BooleanType>>,
    kwargs: FutureRetSpreadKwargs,
) -> Float64Chunked {
    let mut cash = kwargs.init_cash as f64;
    let mut last_pos = 0_f64; // pos_arr[0];
    let mut last_lot_num = 0.;
    if pos_se.is_empty() {
        return Float64Chunked::from_vec(pos_se.name(), vec![]);
    }
    let mut last_close = closing_cost_se.get(0).unwrap();
    let blowup = kwargs.blowup;
    let multiplier = kwargs.multiplier;
    let commision_type = kwargs.commision_type;
    let leverage = kwargs.leverage;
    let c_rate = kwargs.c_rate;
    if let Some(contract_chg_signal_se) = contract_chg_signal_se {
        pos_se
            .into_iter()
            .zip(opening_cost_se.into_no_null_iter())
            .zip(closing_cost_se.into_no_null_iter())
            .zip(spread_se.into_no_null_iter())
            .zip(contract_chg_signal_se.into_no_null_iter())
            .map(
                |((((pos, opening_cost), closing_cost), spread), contract_signal)| {
                    if pos.is_none() {
                        return Some(cash);
                    } else if blowup && cash <= 0. {
                        return Some(0.);
                    }
                    let pos = pos.unwrap();
                    if (last_lot_num != 0.) && (!contract_signal) {
                        // 换月的时候不计算跳开的损益
                        cash += last_lot_num
                            * (opening_cost - last_close)
                            * multiplier
                            * last_pos.signum();
                    }
                    // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                    if (pos != last_pos) || contract_signal {
                        // 仓位出现变化，计算新的理论持仓手数
                        let l =
                            ((cash * leverage * pos.abs()) / (multiplier * opening_cost)).floor();
                        let (lot_num, lot_num_change) = if !contract_signal {
                            (
                                l,
                                (l * pos.signum() - last_lot_num * last_pos.signum()).abs(),
                            )
                        } else {
                            (l, l.abs() * 2.)
                        };
                        // 扣除手续费变动
                        if &commision_type == "percent" {
                            cash -= lot_num_change * multiplier * (opening_cost * c_rate + spread);
                        } else {
                            cash -= lot_num_change * (c_rate + multiplier * spread);
                        };
                        // 更新上期持仓手数和持仓头寸
                        last_lot_num = lot_num;
                        last_pos = pos;
                    }
                    // 计算当期损益
                    if last_lot_num != 0. {
                        cash += last_lot_num
                            * last_pos.signum()
                            * (closing_cost - opening_cost)
                            * multiplier;
                    }
                    last_close = closing_cost; // 更新上期收盘价
                    Some(cash)
                },
            )
            .collect_trusted()
    } else {
        // 不考虑合约换月信号的情况
        pos_se
            .into_iter()
            .zip(opening_cost_se.into_no_null_iter())
            .zip(closing_cost_se.into_no_null_iter())
            .zip(spread_se.into_no_null_iter())
            .map(|(((pos, opening_cost), closing_cost), spread)| {
                if pos.is_none() {
                    return Some(cash);
                } else if blowup && cash <= 0. {
                    return Some(0.);
                }
                let pos = pos.unwrap();
                if last_lot_num != 0. {
                    cash +=
                        last_lot_num * (opening_cost - last_close) * multiplier * last_pos.signum();
                }
                // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                if pos != last_pos {
                    // 仓位出现变化
                    // 计算新的理论持仓手数
                    let lot_num =
                        ((cash * leverage * pos.abs()) / (multiplier * opening_cost)).floor();
                    // 扣除手续费变动
                    if &commision_type == "percent" {
                        cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum()).abs()
                            * multiplier
                            * (opening_cost * c_rate + spread);
                    } else {
                        cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum()).abs()
                            * (c_rate + multiplier * spread);
                    };
                    // 更新上期持仓手数和持仓头寸
                    last_lot_num = lot_num;
                    last_pos = pos;
                }
                // 计算当期损益
                if last_lot_num != 0. {
                    cash += last_lot_num
                        * (closing_cost - opening_cost)
                        * multiplier
                        * last_pos.signum();
                }
                last_close = closing_cost; // 更新上期收盘价
                Some(cash)
            })
            .collect_trusted()
    }
}
