#![allow(clippy::unused_unit)]
use super::StrategyFilter;
use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use serde::Deserialize;

#[derive(Deserialize)]
struct BollVolStopKwargs {
    // window, open_width, stop_width, take_profit_using_return_vol parameter
    params: (usize, f64, f64, f64),
    filter_flag: bool,
    delay_open: bool,
    long_signal: f64,
    short_signal: f64,
    close_signal: f64,
}

#[polars_expr(output_type=Float64)]
fn boll_vol_stop(inputs: &[Series], kwargs: BollVolStopKwargs) -> PolarsResult<Series> {
    let fac = inputs[0].f64()?;
    let middle = inputs[1].f64()?;
    let std_ = inputs[2].f64()?;
    let ret_vol = inputs[3].f64()?;
    let filter = if kwargs.filter_flag {
        Some(StrategyFilter::from_inputs(inputs, (4, 5, 6, 7))?)
    } else {
        None
    };
    Ok(impl_boll_vol_stop(fac, middle, std_, ret_vol, filter, kwargs).into_series())
}

macro_rules! boll_vol_stop_logic_impl {
    (
        $kwargs: expr,
        $fac: expr, $middle: expr, $std: expr, $ret_vol: expr,
        $last_signal: expr, $last_fac: expr, $open_price: expr,
        $(filters=>($long_open: expr, $long_stop: expr, $short_open: expr, $short_stop: expr),)?
        $(long_open=>$long_open_cond: expr,)?
        $(short_open=>$short_open_cond: expr,)?
        $(,)?
    ) => {
        {
            if $fac.is_some() && $middle.is_some() && $std.is_some() && $std.unwrap() > 0. {
                let fac = ($fac.unwrap() - $middle.unwrap()) / $std.unwrap();
                // == open condition
                let mut open_flag = false;
                if ($last_signal != $kwargs.long_signal) && (fac >= $kwargs.params.1) $(&& $long_open.unwrap_or(true))? $(&& $long_open_cond)? {
                    // long open
                    // only update open_price when signal changes
                    $open_price = $fac;
                    $last_signal = $kwargs.long_signal;
                    open_flag = true;
                } else if ($last_signal != $kwargs.short_signal) && (fac <= -$kwargs.params.1) $(&& $short_open.unwrap_or(true))? $(&& $short_open_cond)? {
                    // short open
                    $open_price = $fac;
                    $last_signal = $kwargs.short_signal;
                    open_flag = true;
                }
                // == stop condition
                if (!open_flag) && ($last_signal != $kwargs.close_signal) {
                    // we can skip stop condition if trade is already close or open
                    let mut cond = (($last_fac > $kwargs.params.2) && (fac <= $kwargs.params.2))
                        || ($last_fac < -$kwargs.params.2) && (fac >= -$kwargs.params.2)
                        $(|| $long_stop.unwrap_or(false))?  // additional stop condition
                        $(|| $short_stop.unwrap_or(false))?;
                    if $ret_vol.is_some() && $open_price.is_some() {
                        cond = cond || ($fac.unwrap() >= $open_price.unwrap() + $kwargs.params.3 * $ret_vol.unwrap())
                            || ($fac.unwrap() <= $open_price.unwrap() - $kwargs.params.3 * $ret_vol.unwrap());
                    }
                    if cond {
                        $last_signal = $kwargs.close_signal;
                    }
                }
                // == update open info
                $last_fac = fac;
            }
            Some($last_signal)
        }
    };
}

#[allow(clippy::collapsible_else_if)]
fn impl_boll_vol_stop(
    fac_arr: &Float64Chunked,
    middle_arr: &Float64Chunked,
    std_arr: &Float64Chunked,
    ret_vol_arr: &Float64Chunked,
    filter: Option<StrategyFilter>,
    kwargs: BollVolStopKwargs,
) -> Float64Chunked {
    let m = kwargs.params.1;
    let mut last_signal = kwargs.close_signal;
    let mut last_fac = 0.;
    let mut open_price: Option<f64> = None;
    if let Some(filter) = filter {
        let zip_ = izip!(
            fac_arr,
            middle_arr,
            std_arr,
            ret_vol_arr,
            filter.long_open,
            filter.long_stop,
            filter.short_open,
            filter.short_stop
        );
        if kwargs.delay_open {
            zip_.map(
                |(fac, middle, std, ret_vol, long_open, long_stop, short_open, short_stop)| {
                    boll_vol_stop_logic_impl!(
                        kwargs, fac, middle, std, ret_vol,
                        last_signal, last_fac, open_price,
                        filters=>(long_open, long_stop, short_open, short_stop),
                    )
                },
            )
            .collect_trusted()
        } else {
            zip_.map(
                |(fac, middle, std, ret_vol, long_open, long_stop, short_open, short_stop)| {
                    boll_vol_stop_logic_impl!(
                        kwargs, fac, middle, std, ret_vol,
                        last_signal, last_fac, open_price,
                        filters=>(long_open, long_stop, short_open, short_stop),
                        long_open=>last_fac < m,
                        short_open=>last_fac > -m,
                    )
                },
            )
            .collect_trusted()
        }
    } else {
        if kwargs.delay_open {
            izip!(fac_arr, middle_arr, std_arr, ret_vol_arr)
                .map(|(fac, middle, std, ret_vol)| {
                    boll_vol_stop_logic_impl!(
                        kwargs,
                        fac,
                        middle,
                        std,
                        ret_vol,
                        last_signal,
                        last_fac,
                        open_price,
                    )
                })
                .collect_trusted()
        } else {
            izip!(fac_arr, middle_arr, std_arr, ret_vol_arr)
                .map(|(fac, middle, std, ret_vol)| {
                    boll_vol_stop_logic_impl!(
                        kwargs, fac, middle, std, ret_vol,
                        last_signal, last_fac, open_price,
                        long_open=>last_fac < m,
                        short_open=>last_fac > -m,
                    )
                })
                .collect_trusted()
        }
    }
}
