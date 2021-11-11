from pyspark.ml import Transformer
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark import keyword_only

__all__ = ['LogTransformation', 'LagFeature']


class LogTransformation(Transformer, HasInputCol, HasOutputCol):  # Find the log of given column

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(LogTransformation, self).__init__()
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df):
        return df.withColumn(self.getOutputCol(), log(col(self.getInputCol())))


class LagFeature(Transformer, HasInputCol, HasOutputCol):  # Create lagging columns based on number of features provided

    numberOfFeatures = Param(Params._dummy(), 'numberOfFeatures', 'numberOfFeatures',
                             typeConverter=TypeConverters.toInt)
    time = Param(Params._dummy(), 'time', 'time', typeConverter=TypeConverters.toString)
    partition = Param(Params._dummy(), 'partition', 'partition', typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, numberOfFeatures=None, time=None, partition=None):
        super(LagFeature, self).__init__()
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCol=None, outputCol=None, numberOfFeatures=None, time=None, partition=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def get_numberOfFeature(self):
        return self.getOrDefault(self.numberOfFeatures)

    def getTime(self):
        return self.getOrDefault(self.time)

    def getPartition(self):
        return self.getOrDefault(self.partition)

    def _transform(self, df):
        timeWindow = Window.partitionBy(self.getPartition()).orderBy((self.getTime()))
        for i in range(1, self.get_numberOfFeature() + 1):
            temp = self.getOutputCol() + str(i)
            df = df.withColumn(temp, lag(self.getInputCol(), i).over(timeWindow))
        return df
