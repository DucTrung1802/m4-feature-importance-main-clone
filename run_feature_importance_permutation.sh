#!/bin/bash


FEATURES=TSFresh
FEATURE_FILES="features/$FEATURES.csv"
CORRELATION=0.5
PREDICTIONS=prediction_errors/mape.csv

tsp -S 3

for RUN in {1..30}
do
    for MODEL in M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4KerasNetRegressionModel M4RandomForestSingleOutputRegressor
    do
        tsp python feature_importance_permutation.py --model $MODEL --predictions $PREDICTIONS --features $FEATURE_FILES --feature_correlation_threshold $CORRELATION --run $RUN
    done
done
