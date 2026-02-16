#!/bin/bash

FEATURES=TSFresh
CORRELATION=0.5

for MODEL in M4XGBRegressor M4RandomForestRegressor M4KerasNetRegressionModel M4MLPRegressor M4MLPRegressorLarge M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression
do
    for RUN in {1..10}
    do
        OUTPUT="trained_model/feature_importance_setup/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
        FEATURE_FILES="features/$FEATURES.csv"
        python train_model.py --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT &
    done
done
