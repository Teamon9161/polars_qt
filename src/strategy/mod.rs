mod boll;
mod boll_vol_stop;

use polars::datatypes::BooleanChunked;
use polars::prelude::*;

pub(super) struct StrategyFilter<'a> {
    long_open: &'a BooleanChunked,
    long_stop: &'a BooleanChunked,
    short_open: &'a BooleanChunked,
    short_stop: &'a BooleanChunked,
}

impl<'a> StrategyFilter<'a> {
    fn from_inputs(inputs: &'a [Series], idxs: (usize, usize, usize, usize)) -> PolarsResult<Self> {
        Ok(Self {
            long_open: inputs[idxs.0].bool()?,
            long_stop: inputs[idxs.1].bool()?,
            short_open: inputs[idxs.2].bool()?,
            short_stop: inputs[idxs.3].bool()?,
        })
    }
}
