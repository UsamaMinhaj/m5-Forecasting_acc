from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters, HasInputCols
from pyspark.sql import DataFrame

__all__ = ['ImputeMean', 'MarkZero', 'TrainTestSplit', 'DataAggregation']


class ImputeMean(Transformer, HasInputCols):  # Fill the nulls using the mean of the values in the column

    @keyword_only
    def __init__(self, inputCols=None):
        super(ImputeMean, self).__init__()
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df):
        stats = df.agg(*(
            avg(c).alias(c) for c in self.getInputCols()
        ))
        return df.na.fill(stats.first().asDict())


class MarkZero(Transformer):  # Removes negative and zero values in a column

    def __init__(self):
        super(MarkZero, self).__init__()

    def _transform(self, df):
        return df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])


class ReplaceZero(Transformer, HasInputCol):
    # Replace Zero with custom value in order to prevent Null after log transformation
    @keyword_only
    def __init__(self, inputCol=None):
        super(ReplaceZero, self).__init__()
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df):
        return df.replace(to_replace=0,
                          value=0.5)


class TrainTestSplit(Transformer, HasInputCol):
    ratio = Param(Params._dummy(),
                  "ratio", "ratio",
                  typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, inputCol=None, ratio=0.7):
        super(TrainTestSplit, self).__init__()
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None, ratio=0.7):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df):
        # Splitting the time series data using ranking by date so that future values are not used for training past data
        df = df.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy(self.getInputCol())))
        train = df.where('rank <= 0.8').drop('rank')
        test = df.where('rank > 0.8').drop('rank')
        return train, test


class DataAggregation(Transformer, HasInputCol, HasOutputCol):  # Data aggregation on store-department level
    group = Param(Params._dummy(), 'group', 'group name', typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, group=None):
        super(DataAggregation, self).__init__()
        self.group = Param(self, "group", "")
        self._setDefault(group=[])
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None, outputCol=None, group=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getGroup(self):
        return self.getOrDefault(self.group)

    def _transform(self, df: DataFrame) -> DataFrame:
        print(self.group)
        return df.groupBy(self.getGroup()).agg(sum(self.getInputCol()).alias(self.getOutputCol()))
