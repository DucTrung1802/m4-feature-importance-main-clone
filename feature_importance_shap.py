from utils import *
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import argparse
from models import *
import os
import sys
from collections import defaultdict
from os.path import exists
import shap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True, choices=list(M4Model.get_registered_models().keys()))
    parser.add_argument('--run', type=int, required=False, default=0)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--feature_correlation_threshold', type=float, required=False, default=1)
    args = parser.parse_args()
    
    fdf = get_problem_features(args.features)
    pdf = get_problem_algorithm_performance(args.predictions)
    fdf = fdf.pipe(drop_nan_and_non_unique_columns)

    X_train, X_test, Y_train, Y_test = train_test_split(fdf, pdf, test_size=0.1, random_state=args.run)

    if args.feature_correlation_threshold<1.0:
        shape_before = X_train.shape
        drop_columns = set(find_correlation(X_train, thresh=args.feature_correlation_threshold))
        X_train = X_train.drop(columns=drop_columns)
        X_test = X_test.drop(columns=drop_columns)
        shape_after = X_train.shape
        print(f'feature_correlation_threshold ({args.feature_correlation_threshold}) changed shape from {shape_before} to {shape_after}')

    used_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model_type = M4Model.get_registered_model(args.model)
    model_load_dir = f'trained_model/feature_compare/m_{args.model}_f_TSFresh_r_{args.run}_c_{args.feature_correlation_threshold}'
    model = model_type.load_model(model_load_dir)
    
    
    feature_importance_dir = 'feature_importance'
    shap_raw_dir = 'shap_raw'
        
    
    
    for i_column, column in enumerate(list(Y_train.columns)):
        feature_importance_file = f'{feature_importance_dir}/r_{args.run}_m_{args.model}_f_shap-{column}.p'
        shap_raw_file = f'{shap_raw_dir}/r_{args.run}_m_{args.model}_f_shap-{column}.p'
        
        if exists(feature_importance_file):
            print('Skip computing. File exists')
            continue
            
        #if column not in ['118', '245', '237', '72', '69', '36', '78', 'Theta', 'Com', 'ARIMA', 'Damped', 'ETS', 'Naive2', 'Naive', 'sNaive','RNN']:
        #    print('Skip computing. Forecasting algortihm not included')
        #    continue
        
        emb = {}
        
        model.set_single_predict(i_column)

        kmeans_size = 10
        X_train_summary = shap.kmeans(X_train, kmeans_size)

        ex = shap.KernelExplainer(model.predict, X_train_summary)

        shap_values = ex.shap_values(X_test[:512, :], nsamples=1024)

        vals = np.abs(shap_values).mean(0)

        emb[column] = list(zip(vals, used_columns))

        create_directory_if_not_exist(feature_importance_dir)
        pickle.dump(emb, open(feature_importance_file, "wb"))
        
        create_directory_if_not_exist(shap_raw_dir)
        pickle.dump(shap_values, open(shap_raw_file, "wb"))
        



