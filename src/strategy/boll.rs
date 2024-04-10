#![allow(clippy::unused_unit)]
use super::StrategyFilter;
use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use serde::Deserialize;

#[derive(Deserialize)]
struct BollKwargs {
    params: (usize, f64, f64, Option<f64>),
    filter_flag: bool,
    delay_open: bool,
    long_signal: f64,
    short_signal: f64,
    close_signal: f64,
}

#[polars_expr(output_type=Float64)]
fn boll(inputs: &[Series], kwargs: BollKwargs) -> PolarsResult<Series> {
    let fac = inputs[0].f64()?;
    let middle = inputs[1].f64()?;
    let std = inputs[2].f64()?;
    let filter = if kwargs.filter_flag {
        Some(StrategyFilter::from_inputs(inputs, (3, 4, 5, 6))?)
    } else {
        None
    };
    Ok(impl_boll(fac, middle, std, filter, kwargs).into_series())
}

macro_rules! boll_logic_impl {
    (
        $kwargs: expr,
        $fac: expr, $middle: expr, $std: expr,
        $last_signal: expr, $last_fac: expr,
        $(filters=>($long_open: expr, $long_stop: expr, $short_open: expr, $short_stop: expr),)?
        $(long_open=>$long_open_cond: expr,)?
        $(short_open=>$short_open_cond: expr,)?
        $(profit_p=>$m3: expr)?
        $(,)?
    ) => {
        {
            if $fac.is_some() && $middle.is_some() && $std.is_some() && $std.unwrap() > 0. {
                let fac = ($fac.unwrap() - $middle.unwrap()) / $std.unwrap();
                // == open condition
                let mut open_flag = false;
                if (fac >= $kwargs.params.1) $(&& $long_open.unwrap_or(true))? $(&& $long_open_cond)? {
                    // long open
                    $last_signal = $kwargs.long_signal;
                    open_flag = true;
                } else if (fac <= -$kwargs.params.1) $(&& $short_open.unwrap_or(true))? $(&& $short_open_cond)? {
                    // short open
                    $last_signal = $kwargs.short_signal;
                    open_flag = true;
                }
                // == stop condition
                if (!open_flag) && ($last_signal != $kwargs.close_signal) {
                    // we can skip stop condition if trade is already close or open
                    if ($last_fac > $kwargs.params.2) && (fac <= $kwargs.params.2)
                        $(|| $long_stop.unwrap_or(false))?  // additional stop condition
                        $(|| fac > $m3)?  // profit stop condition
                    {
                        // long stop
                        $last_signal = $kwargs.close_signal;
                    } else if ($last_fac < -$kwargs.params.2) && (fac >= -$kwargs.params.2)
                        $(|| $short_stop.unwrap_or(false))?
                        $(|| fac < -$m3)?  // profit stop condition
                    {
                        // short stop
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
fn impl_boll(
    fac_arr: &Float64Chunked,
    middle_arr: &Float64Chunked,
    std_arr: &Float64Chunked,
    filter: Option<StrategyFilter>,
    kwargs: BollKwargs,
) -> Float64Chunked {
    let m = kwargs.params.1;
    let mut last_signal = kwargs.close_signal;
    let mut last_fac = 0.;
    if let Some(filter) = filter {
        let zip_ = izip!(
            fac_arr,
            middle_arr,
            std_arr,
            filter.long_open,
            filter.long_stop,
            filter.short_open,
            filter.short_stop
        );
        if kwargs.delay_open {
            if let Some(m3) = kwargs.params.3 {
                zip_.map(
                    |(fac, middle, std, long_open, long_stop, short_open, short_stop)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            filters=>(long_open, long_stop, short_open, short_stop),
                            profit_p=>m3,
                        )
                    },
                )
                .collect_trusted()
            } else {
                zip_.map(
                    |(fac, middle, std, long_open, long_stop, short_open, short_stop)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            filters=>(long_open, long_stop, short_open, short_stop),
                        )
                    },
                )
                .collect_trusted()
            }
        } else {
            if let Some(m3) = kwargs.params.3 {
                zip_.map(
                    |(fac, middle, std, long_open, long_stop, short_open, short_stop)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            filters=>(long_open, long_stop, short_open, short_stop),
                            long_open=>last_fac < m,
                            short_open=>last_fac > -m,
                            profit_p=>m3,
                        )
                    },
                )
                .collect_trusted()
            } else {
                zip_.map(
                    |(fac, middle, std, long_open, long_stop, short_open, short_stop)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            filters=>(long_open, long_stop, short_open, short_stop),
                            long_open=>last_fac < m,
                            short_open=>last_fac > -m,
                        )
                    },
                )
                .collect_trusted()
            }
        }
    } else {
        if kwargs.delay_open {
            if let Some(m3) = kwargs.params.3 {
                izip!(fac_arr, middle_arr, std_arr)
                    .map(|(fac, middle, std)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            profit_p=>m3,
                        )
                    })
                    .collect_trusted()
            } else {
                izip!(fac_arr, middle_arr, std_arr)
                    .map(|(fac, middle, std)| {
                        boll_logic_impl!(kwargs, fac, middle, std, last_signal, last_fac,)
                    })
                    .collect_trusted()
            }
        } else {
            if let Some(m3) = kwargs.params.3 {
                izip!(fac_arr, middle_arr, std_arr)
                    .map(|(fac, middle, std)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            long_open=>last_fac < m,
                            short_open=>last_fac > -m,
                            profit_p=>m3,
                        )
                    })
                    .collect_trusted()
            } else {
                izip!(fac_arr, middle_arr, std_arr)
                    .map(|(fac, middle, std)| {
                        boll_logic_impl!(
                            kwargs, fac, middle, std,
                            last_signal, last_fac,
                            long_open=>last_fac < m,
                            short_open=>last_fac > -m,
                            // long_open=>|last_fac| (last_fac < m),
                            // short_open=>|last_fac| (last_fac > -m),
                        )
                    })
                    .collect_trusted()
            }
        }
    }
}
