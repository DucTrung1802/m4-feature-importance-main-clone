#!/bin/bash

#MODEL=M4DecisionTreeRegressor
#MODEL=M4RandomForestRegressor

MODEL=M4DummyMeanRegression
MODEL=M4MLPRegressor

FEATURES=TSFresh
FEATURE_FILES="features/$FEATURES.csv"
CORRELATION=0.5
PREDICTIONS=prediction_errors/mape.csv

for RUN in {1..6}
do
    OUTPUT="feature_importance/one_feature/m_${MODEL}_f_${FEATURES}_r_${RUN}"
    python one_feature_importance.py --model $MODEL --predictions $PREDICTIONS --features $FEATURE_FILES --feature_correlation_threshold $CORRELATION --run $RUN --output_save $OUTPUT &
done

#FEATURES=Catch22
#
#for MODEL in M4MLPRegressor M4MLPRegressorLarge M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4DummyMedianRegression
#do
#    for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
#    do
#        for RUN in {1..6}
#        do
#            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
#            FEATURE_FILES="features/$FEATURES.csv"
#            python train_model.py --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT &
#        done
#    done
#done
#
#
#FEATURES=TSFresh
#
#for MODEL in M4MLPRegressor M4MLPRegressorLarge M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4DummyMedianRegression
#do
#    for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
#    do
#        for RUN in {1..6}
#        do
#            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES}_r_${RUN}_c_${CORRELATION}"
#            FEATURE_FILES="features/$FEATURES.csv"
#            python train_model.py --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT &
#        done
#    done
#done
#
#
#FEATURES1=TSFresh
#FEATURES2=Catch22
#
#for MODEL in M4MLPRegressor M4MLPRegressorLarge M4KNeighborsRegressor M4KNeighborsCosineRegressor M4DummyMeanRegression M4DummyMedianRegression
#do
#    for CORRELATION in 1.0 0.95 0.9 0.8 0.7 0.6 0.5
#    do
#        for RUN in {1..6}
#        do
#            OUTPUT="trained_model/feature_compare/m_${MODEL}_f_${FEATURES1},${FEATURES2}_r_${RUN}_c_${CORRELATION}"
#            FEATURE_FILES="features/$FEATURES1.csv features/$FEATURES2.csv"
#            python train_model.py --model $MODEL --run $RUN --predictions prediction_errors/mape.csv --feature_correlation_threshold $CORRELATION --features $FEATURE_FILES --output_save $OUTPUT &
#        done
#    done
#done