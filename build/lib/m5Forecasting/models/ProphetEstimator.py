from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, Param, Params, TypeConverters, HasLabelCol
from pyspark.ml import Estimator

from pyspark.sql.types import StructField, StructType, StringType, LongType, DoubleType, DateType
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from fbprophet import Prophet
from statsmodels.tools.eval_measures import rmse

__all__ = ['ProphetEstimator']


class ProphetEstimator(Estimator, HasLabelCol, HasPredictionCol):
    group = Param(Params._dummy(), 'group', 'group name', typeConverter=TypeConverters.toListString)
    cutoff = Param(Params._dummy(), 'cutoff', 'Date cutoff', typeConverter=TypeConverters.toString)

    schema = StructType([
        StructField("store_id", StringType(), True),
        StructField("item_id", StringType(), True),
        StructField("ds", DateType(), True),
        StructField("yhat", DoubleType(), True)
    ])

    @keyword_only
    def __init__(self, labelCol=None, predictionCol=None, cutoff=None, group=None):
        super(ProphetEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self.model = None  # Fitted FB prophet model
        self.rmse_fb = None

    @keyword_only
    def setParams(self, labelCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getCutoff(self):
        return self.getOrDefault(self.cutoff)

    def getGroup(self):
        return self.getOrDefault(self.group)

    def getRmse(self):
        return self.rmse_fb

    def _fit(self, dataset):
        predictions = dataset.groupBy(self.getGroup()).apply(self.fit_pandas_udf)
        self.rmse_fb = rmse(predictions.toPandas()['yhat'],
                            dataset.filter(dataset.date > self.getCutoff()).toPandas()['sales'])
        return ProphetEstimatorModel(trainedModel=self.model)

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def fit_pandas_udf(self, df):
        """
        :param df: Dataframe (train + test data)
        :return: predictions as defined in the output schema
        """

        def train_fitted_prophet(df, cutoff):
            # train
            ts_train = (df
                        .query('date <= @cutoff')
                        .rename(columns={'date': 'ds', 'sales': 'y'})
                        .sort_values('ds')
                        )
            ts_test = (df
                       .query('date > @cutoff')
                       .rename(columns={'date': 'ds', 'sales': 'y'})
                       .sort_values('ds')
                       .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                       .drop('y', axis=1)
                       )

            m = Prophet(yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True)

            m.fit(ts_train)

            df["date"] = pd.to_datetime(df["date"])
            # at this step we predict the future and we get plenty of additional columns be cautious
            ts_hat = (m.predict(ts_test)[["ds", "yhat"]]
                      .assign(ds=lambda x: pd.to_datetime(x["ds"]))
                      ).merge(ts_test, on=["ds"], how="left")  # merge to retrieve item and store index

            return pd.DataFrame(ts_hat, columns=self.schema.fieldNames()), m

        df, self.model = train_fitted_prophet(df, self.getCutoff())
        return df


class ProphetEstimatorModel(Model):
    def __init__(self, trainedModel=None):
        super(ProphetEstimatorModel, self).__init__()
        self.trainedModel = trainedModel

    def _transform(self, dataset):
        return self.trainedModel.transform(dataset)