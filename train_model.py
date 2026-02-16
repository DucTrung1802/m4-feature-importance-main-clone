from utils import *
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from models import *
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True, choices=list(M4Model.get_registered_models().keys()))
    parser.add_argument('--run', type=int, required=False, default=0)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--feature_correlation_threshold', type=float, required=False, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    
    parser.add_argument('--output_save', type=str, required=True)
    
    args = parser.parse_args()
    
    if os.path.isdir(args.output_save):
        print("Model trained. Skipping...")
    else:
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
        
        scalerY = MinMaxScaler()
        Y_train = scalerY.fit_transform(Y_train)

        model = M4Model.get_registered_model(args.model)(seed=args.run)
        model.fit(X_train, Y_train)

        if args.save_model > 0:
            model.save(args.output_save)
        
        pred = scalerY.inverse_transform(model.predict(X_test))
        
        error_mae = mean_absolute_error(pred, Y_test)
        error_mse = mean_squared_error(pred, Y_test)
        
        error_column_mae = mean_absolute_error(pred, Y_test, multioutput='raw_values')
        error_column_mse = mean_squared_error(pred, Y_test, multioutput='raw_values')
        
        save_meta = {'error_mae': error_mae, 
                     'error_mse': error_mse,
                     'used_columns': used_columns, 
                     'error_column_mae': error_column_mae, 
                     'error_column_mse': error_column_mse}
        
        create_directory_if_not_exist(args.output_save)
        pickle.dump(save_meta, open(f'{args.output_save}/meta.p', 'wb'))
