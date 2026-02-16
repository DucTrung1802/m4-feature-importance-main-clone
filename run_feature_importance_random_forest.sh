#!/bin/bash

FEATURES=TSFresh
FEATURE_FILES="features/$FEATURES.csv"
CORRELATION=0.5
PREDICTIONS=prediction_errors/mape.csv

tsp -S 20

for RUN in {1..30}
do
    tsp python feature_importance_random_forest.py --predictions $PREDICTIONS --features $FEATURE_FILES --feature_correlation_threshold $CORRELATION --run $RUN
done
