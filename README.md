# M4 Feature Importance

Accurate and reliable forecasting is a crucial task in many different domains. The selection of a forecasting algorithm that is suitable for a specific time series can be a challenging task, since the algorithms' performance depends on the time-series properties, as well as the properties of the forecasting algorithms. This paper addresses two aspects of forecasting algorithm performance prediction. Firstly, we explore the possibility of applying meta-learning to predict the performance of a forecasting algorithm, given time-series features extracted using the tsfresh and catch22 libraries. We analyze the predictive capacity of these features using a variety of feature importance methods. Our results indicate that different machine learning models rely on different features in order to predict the performance of a forecasting algorithm, and that different feature importance methods assign different importance values to features, even when the same meta-models are used for predicting the performance of the same forecasting algorithm. We show that there are connections between time-series properties and forecasting algorithms and that certain algorithms are highly dependent on the properties of the time-series, which further provides more insight into understanding the algorithm's performance.

## Usage

### Build the image

docker-compose up --build --force-recreate -d

### Inside docker run

1. `cd work` - Move to the working directory inside Docker container
2. `conda activate Base` - Activate Base environemnt
3. `./run_algorithms_tsfresh_vs_catch22.sh` - Train all models. This script runs other Python scripts for training (Not that this might takes weeks)
4. `./run_feature_importance_permutation.sh` - Compute permutation feature importance
5. `./run_feature_importance_random_forest.sh` - Compute random forest feature importance
6. `./run_feature_importance_shap.sh` - Compute SHAP feature importance
7. `./run_feature_importance_xgboost.sh` - Compute XGBoost feature importance
8. `python create_feature_importance_csv.py` - Generate feature importance CSV file used for the analysis

### Notebooks

Notebooks `00xx-name.ipynb` are used to analyize the results
