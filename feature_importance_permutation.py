from utils import *
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    output_columns = list(Y_train.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    scalerY = MinMaxScaler()
    Y_train = scalerY.fit_transform(Y_train)
    Y_test = scalerY.transform(Y_test)
    
    model_type = M4Model.get_registered_model(args.model)
    model_load_dir = f'trained_model/feature_compare/m_{args.model}_f_TSFresh_r_{args.run}_c_{args.feature_correlation_threshold}'
    model = model_type.load_model(model_load_dir)
    
    feature_importance_dir = 'feature_importance'
    feature_importance_file = f'{feature_importance_dir}/r_{args.run}_m_{args.model}_f_permutation.p'

    if exists(feature_importance_file)==False:
        
        emb = {}
        for i_column, column in enumerate(output_columns):
            model.set_single_predict(i_column)
            r = permutation_importance(model, X_test, Y_test[:, i_column], n_repeats=5, random_state=args.run, scoring='neg_mean_squared_error')
            l = []
            for i in r.importances_mean.argsort()[::-1]:
                l.append((r.importances_mean[i], used_columns[i]))
            emb[column] = l

        create_directory_if_not_exist(feature_importance_dir)
        pickle.dump(emb, open(feature_importance_file, "wb"))
    else:
        print('Permutation already exists')


