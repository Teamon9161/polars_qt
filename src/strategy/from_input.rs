use polars::prelude::*;
use tea_strategy::StrategyFilter;

pub trait FromInput<'a> {
    fn from_inputs(inputs: &'a [Series], idxs: &'a [usize]) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl<'a> FromInput<'a> for StrategyFilter<&'a BooleanChunked> {
    fn from_inputs(inputs: &'a [Series], idxs: &'a [usize]) -> PolarsResult<Self> {
        Ok(Self {
            long_open: inputs[idxs[0]].bool()?,
            long_stop: inputs[idxs[1]].bool()?,
            short_open: inputs[idxs[2]].bool()?,
            short_stop: inputs[idxs[3]].bool()?,
        })
    }
}
