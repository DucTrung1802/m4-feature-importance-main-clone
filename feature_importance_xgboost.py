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
from sklearn.inspection import permutation_importance
from os.path import exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
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

    # No need to transform the data since we are just reading importance values
    
    model_name = 'M4XGBRegressor'
    
    model_type = M4Model.get_registered_model(model_name)
    model_load_dir = f'trained_model/feature_compare/m_{model_name}_f_TSFresh_r_{args.run}_c_{args.feature_correlation_threshold}'
    model = model_type.load_model(model_load_dir)
    
    feature_importance_dir = 'feature_importance'
    
    for feature_importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
    
        feature_importance_file = f'{feature_importance_dir}/r_{args.run}_m_{model_name}_f_xgboost-{feature_importance_type}.p'

        if exists(feature_importance_file)==False:        
            emb = {}

            for i_column, column in enumerate(list(Y_train.columns)):
                
                d = model.model.estimators_[i_column].get_booster().get_score(importance_type=feature_importance_type)
                l = []
                for i, c in enumerate(used_columns):
                    k = f'f{i}'
                    if k in d.keys():
                        l.append((d[k], c))
                    else:
                        l.append((0.0, c))
                emb[column] = l

            create_directory_if_not_exist(feature_importance_dir)
            pickle.dump(emb, open(feature_importance_file, "wb"))
        else:
            print(f'XGBoost {feature_importance_type} importance already exists')


