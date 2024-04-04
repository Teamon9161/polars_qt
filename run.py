import polars as pl
import polars_qt as pq

df = pl.DataFrame({
    # 'a': [1, 1, None],
    'b': [5.2, 4.1, 6.3, None, 10, 4, 5],
    # 'c': ['hello', 'everybody!', '!']
})
print(df.with_columns(
    pq.rolling_rank(pl.col('b'), 4, pct=True).alias('1'),
    pl.col('b').qt.rolling_rank(4, pct=False).alias('2')
))
