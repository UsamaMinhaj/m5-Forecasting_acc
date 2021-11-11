import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from typing import Iterable
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import IntegerType
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from ForecastingPipelineUsama.PreProcessing import MarkZero, ImputeMean, TrainTestSplit, DataAggregation, ReplaceZero
from ForecastingPipelineUsama.FeatureEngineering import LagFeature, LogTransformation
from ForecastingPipelineUsama.HyperOptimization import XgbEstimator, ProphetEstimator

# from ForecastingPipelineUsama.PreProcessing import DataAggregation
import numpy as np


def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                        if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']


# Custom implementation of melt from pyspark (from stackoverflow)
def melt(
        df: DataFrame,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str = "variable", value_name: str = "value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))

    cols = id_vars + [
        col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


if __name__ == '__main__':
    spark = SparkSession.builder.master("local[3]").appName("SparkByExample.com").getOrCreate()
    mlflow.pyspark.ml.autolog()

    # Load all the data files
    calendarDF = spark.read.csv('calendar.csv', inferSchema=True, header=True)
    validationDF = spark.read.csv('sales_train_validation.csv', inferSchema=True,
                                  header=True)
    evaluationDF = spark.read.csv('sales_train_evaluation.csv', inferSchema=True,
                                  header=True)
    sampleDF = spark.read.csv('sample_submission.csv', inferSchema=True,
                              header=True)
    sellPricesDF = spark.read.csv('sell_prices.csv', inferSchema=True,
                                  header=True)

    # Get the id of sales by concatenating item_id and store_id
    sellPricesDF2 = sellPricesDF.withColumn('id', concat(col('item_id'), lit('_'), col('store_id'), lit('_evaluation')))

    calendar2 = sellPricesDF2.join(calendarDF, 'wm_yr_wk',
                                   'left')  # Joining sellPrices with Calendar to get info about each sales day
    spark.conf.set('spark.sql.autoBroadcastJoinThreshold', '-1')

    # Converting days column into rows in order to join with other tables
    df_melted = melt(evaluationDF, id_vars=['id'], value_vars=evaluationDF.columns[6:])

    # Getting day column as integer type
    day_conv = df_melted.withColumn('day', regexp_replace('variable', 'd_', '').cast(IntegerType()))
    day_conv = day_conv.withColumnRenamed('id', 'id2')

    df_merged = day_conv.join(calendar2, (calendar2.d == day_conv.variable) & (calendar2.id == day_conv.id2),
                              'inner')  # Final join to merge all tables

    df_merged = df_merged.drop('variable').drop('id2')  # Dropping extra columns after inner join
    df_merged = df_merged.withColumnRenamed('value', 'quantity')
    df_merged = df_merged.withColumn('sales', col('quantity') * col('sell_price'))  # Getting sales column

    # Adding dept_id to the df using item_id
    split_col = split(df_merged['item_id'], '_')
    df_merged = df_merged.withColumn('dept_id', concat(split_col.getItem(0), lit('_'), split_col.getItem(1)))

    DA = DataAggregation(inputCol='sales', outputCol='sales',
                         group=['store_id', 'dept_id', 'day', 'date', 'weekday', 'wday', 'month', 'year', 'd',
                                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',
                                'snap_WI'])
    MK = MarkZero()
    RZ = ReplaceZero(inputCol='sales')  # Replace with custom value to prevent Null values after log transformation

    # Preprocessing Pipeline
    preprocessingPipeline = Pipeline(stages=[DA, RZ])
    print('before error')
    df_preprocessed = preprocessingPipeline.fit(df_merged).transform(df_merged)

    logPy = LogTransformation(inputCol='sales', outputCol='LogSales')
    Lf = LagFeature(inputCol='LogSales', outputCol='LagFeatures', numberOfFeatures=12, time='day',
                    partition=['dept_id', 'store_id'])
    IM = ImputeMean(
        inputCols=['LagFeatures1', 'LagFeatures2', 'LagFeatures3', 'LagFeatures4', 'LagFeatures5', 'LagFeatures6',
                   'LagFeatures7', 'LagFeatures8', 'LagFeatures9', 'LagFeatures10', 'LagFeatures11', 'LagFeatures12'])

    # Feature Engineering Pipeline
    featurePipeline = Pipeline(stages=[logPy, Lf, IM])

    df_featured = (featurePipeline.fit(df_preprocessed).transform(df_preprocessed))

    string_list = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'year', 'dept_id', 'store_id']
    one_hot_list = ['event_name_1', 'event_type_1']
    id2 = [x + '_indexed' for x in string_list]

    StrInd = StringIndexer(inputCols=string_list, outputCols=id2, handleInvalid='keep')

    id3 = [x + '_encoded' for x in one_hot_list]
    one_hot_encoder = OneHotEncoder(inputCols=id2[0:2], outputCols=id3)

    # Encoding Pipeline for columns with String
    encodingPipeline = Pipeline(stages=[StrInd, one_hot_encoder])
    df_encoded = encodingPipeline.fit(df_featured).transform(df_featured)

    # Dropping all useless (repeated) columns
    df_encoded = df_encoded.drop(*id2[:-2], *string_list, 'date', 'd', 'weekday', 'sales')
    featuresCols = df_encoded.columns
    featuresCols.remove('LogSales')

    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
    df_engineered = vectorAssembler.transform(df_encoded)

    # Train Test Split using ranking for time series data
    TTF = TrainTestSplit(inputCol='day', ratio=0.7)
    train, test = TTF.transform(df_engineered)

    # Parameter Space for hyperparameters optimization
    space = {
        'max_depth': hp.choice('max_depth', [2, 5]),
        'n_estimators': hp.choice('n_estimators', [10, 100])
    }
    algo = tpe.suggest

    xgb_regressor = XgbEstimator(labelCol="LogSales", missing=0.0, max_depth=2,
                                 n_estimators=10)
    trials = Trials()
    fmin_obj = partial(xgb_regressor.train_with_hyperopt, train=train, test=test)

    # Optimization using hyperopt
    with mlflow.start_run():
        best_params = fmin(
            fn=fmin_obj,
            space=space,
            algo=algo,
            max_evals=8,
            trials=trials
        )
    rmse_xgb = min(trials.losses())

    # Facebook Prophet
    cutoff = '2016-01-29'
    df_merged_prophet = df_merged.select('store_id', 'item_id', 'sales', 'date')  # Only two columns for prediction
    FBP_estimator = ProphetEstimator(cutoff=cutoff, group=['store_id', 'item_id'])
    fbpModel = FBP_estimator.fit(df_merged_prophet)
    rmse_fb = FBP_estimator.getRmse()

    # Model Selection
    if rmse_xgb < rmse_fb:
        print('XGB performs better')
        bestModel = getBestModelfromTrials(trials)
    else:
        print('FB prophet is the best model')
        bestModel = fbpModel
