mod from_input;
#[macro_use]
mod macros;

use crate::define_strategy;
use from_input::FromInput;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use tea_strategy::*;

define_strategy!(boll, BollKwargs);
define_strategy!(auto_boll{?}, AutoBollKwargs);
define_strategy!(delay_boll{?}, DelayBollKwargs);
define_strategy!(martingale{?}, MartingaleKwargs);
define_strategy!(fix_time{?}, FixTimeKwargs);
define_strategy!(auto_tangqian{?}, AutoTangQiAnKwargs);
define_strategy!(prob_threshold{?}, ProbThresholdKwargs);
