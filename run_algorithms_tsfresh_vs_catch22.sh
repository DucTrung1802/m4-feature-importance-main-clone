#!/bin/bash

tsp -S 3

for RUN in {1..30}
do
    FEATURES=TSFresh

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        for CORRELATION in 0.5
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES.csv"
            tsp python train_model.py --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT  
        done
    done
done


for RUN in {1..30}
do

    FEATURES=Catch22

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES.csv"
            tsp python train_model.py --save_model 0 --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT
        done
    done


    FEATURES=TSFresh

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        # Missing 0.5
        for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES.csv"
            tsp python train_model.py --save_model 0 --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT  
        done
    done


    FEATURES1=TSFresh
    FEATURES2=Catch22

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES1},${FEATURES2}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES1.csv features/$FEATURES2.csv"
            tsp python train_model.py --save_model 0 --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT
        done
    done
    
    
    FEATURES1=Catch22
    FEATURES2=Catch22Diff
    FEATURES3=Catch22Log

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES1},${FEATURES2},${FEATURES3}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES1.csv features/$FEATURES2.csv features/$FEATURES3.csv"
            tsp python train_model.py --save_model 0 --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT
        done
    done
    
    
    FEATURES1=TSFresh
    FEATURES2=TSFreshDiff
    FEATURES3=TSFreshLog

    for MODEL in M4KerasNetRegressionModel M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4XGBRegressor M4RandomForestRegressor M4RandomForestSingleOutputRegressor
    do
        for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
        do
            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES1},${FEATURES2},${FEATURES3}_r_${RUN}_c_${CORRELATION}"
            FEATURE_FILES="features/$FEATURES1.csv features/$FEATURES2.csv features/$FEATURES3.csv"
            tsp python train_model.py --save_model 0 --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT
        done
    done
done