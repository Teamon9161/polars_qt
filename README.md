# polars_qt
[![PyPI](https://img.shields.io/pypi/v/polars_qt)](https://pypi.org/project/polars_qt)

Useful Quant expressions for polars implemented by polars plugin.

Currently :

* Rolling_rank expression

* If-Then expression.

* Calculate return of future using strategy signal


### rolling_rank

```python
import polars as pl
import polars_qt as pq
df = pl.DataFrame({
    'a': [5.2, 4.1, 6.3, None, 10, 4, 5],
})
df.with_columns(
    pq.rolling_rank(pl.col('a'), 4, min_periods=1, pct=True).alias('a_rank'),
    pl.col('a').qt.rolling_rank(4, pct=False, rev=True).alias('a_rank2')
)

shape: (7, 3)
┌──────┬──────────┬─────────┐
│ a    ┆ a_rank   ┆ a_rank2 │
│ ---  ┆ ---      ┆ ---     │
│ f64  ┆ f64      ┆ f64     │
╞══════╪══════════╪═════════╡
│ 5.2  ┆ 1.0      ┆ null    │
│ 4.1  ┆ 0.5      ┆ 2.0     │
│ 6.3  ┆ 1.0      ┆ 1.0     │
│ null ┆ null     ┆ null    │
│ 10.0 ┆ 1.0      ┆ 1.0     │
│ 4.0  ┆ 0.333333 ┆ 3.0     │
│ 5.0  ┆ 0.666667 ┆ 2.0     │
└──────┴──────────┴─────────┘
```



### If-then

```python
df = pl.DataFrame({
    'g': ['a', 'a', 'b', 'a', 'b'],
    'v': [1, 3, 5, 2, 4],
})
df.select(pl.col('v').qt.if_then((pl.len()>2), pl.col('v')*2).over('g'))

shape: (5, 1)
┌─────┐
│ v   │
│ --- │
│ i64 │
╞═════╡
│ 2   │
│ 6   │
│ 5   │
│ 4   │
│ 4   │
└─────┘
```



适用于金融量化领域的polars表达式扩展，使用polars plugin实现。

目前支持：
* 滚动排序
* if_then表达式
* 利用策略信号回测收益