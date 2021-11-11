from hyperopt import STATUS_OK
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, Param, Params, TypeConverters, HasLabelCol
from pyspark.ml import Pipeline, Estimator

from sparkdl.xgboost import XgboostRegressor

__all__ = ['XgbEstimator']

class XgbEstimator(Estimator, HasLabelCol, HasPredictionCol):
    n_estimators = Param(Params._dummy(),
                         "n_estimators", "n_estimators",
                         typeConverter=TypeConverters.toInt)

    missing = Param(Params._dummy(),
                    "missing", "missing",
                    typeConverter=TypeConverters.toInt)

    max_depth = Param(Params._dummy(),
                      "max_depth", "max_depth",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, labelCol=None, predictionCol=None,
                 missing=None, max_depth=None,
                 n_estimators=None):
        super(XgbEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, predictionCol=None,
                  missing=None, max_depth=None,
                  n_estimators=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def get_n_estimators(self):
        return self.getOrDefault(self.n_estimators)

    def get_max_depth(self):
        return self.getOrDefault(self.max_depth)

    def get_missing(self):
        return self.getOrDefault(self.missing)

    def _fit(self, dataset):
        trainedModel = self.xgb_regressor.fit(dataset)
        return XgbEstimatorModel(trainedModel=trainedModel)

    def train_with_hyperopt(self, params, train, test):
        """
        :param test: testing data
        :param train:
        :param params: hyperparameters as a dict. Its structure is consistent with how search space is defined. See below.
        :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
        """
        # For integer parameters, make sure to convert them to int type if Hyperopt is searching over a continuous range of values.
        max_depth = int(params['max_depth'])
        n_estimators = int(params['n_estimators'])

        model, rmse = self.train_tree(max_depth, n_estimators, train, test)

        loss = rmse
        return {'loss': loss, 'status': STATUS_OK}

    def train_tree(self, max_depth, n_estimators, train, test):
        self.xgb_regressor = XgboostRegressor(num_workers=3, labelCol=self.getLabelCol(), missing=0.0,
                                              max_depth=max_depth,
                                              n_estimators=n_estimators)

        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol=self.xgb_regressor.getLabelCol(),
                                        predictionCol=self.xgb_regressor.getPredictionCol())

        pipelineModel = self.xgb_regressor.fit(train)
        predictions = pipelineModel.transform(test)
        rmse = evaluator.evaluate(predictions)
        print("RMSE on our test set: %g" % rmse)
        return pipelineModel, rmse


class XgbEstimatorModel(Model):
    def __init__(self, trainedModel=None):
        super(XgbEstimatorModel, self).__init__()
        self.trainedModel = trainedModel

    def _transform(self, dataset):
        return self.trainedModel.transform(dataset)
