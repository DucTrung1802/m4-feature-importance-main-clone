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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True, choices=list(M4Model.get_registered_models().keys()))
    parser.add_argument('--run', type=int, required=False, default=0)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--feature_correlation_threshold', type=float, required=False, default=1)

    parser.add_argument('--output_save', type=str, required=True)
    
    args = parser.parse_args()
    
    model_repeats = 5
    
    #if os.path.isdir(args.output_save):
    #    print("Model trained. Skipping...")
    #else:
    #    print("Feature importance")
    
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

    perf = defaultdict(list)

    for ifeature, feature in enumerate(used_columns):
        feature_performance = []
        for i in range(model_repeats):
            model_type = M4Model.get_registered_model(args.model)

            model_save_dir = f'{args.output_save}/f_{feature}_i_{i}'
            try:
                model = model_type.load_model(model_save_dir)
            except:
                model = model_type(seed=None)
                model.fit(X_train[:, ifeature].reshape(-1, 1), Y_train)
                model.save(model_save_dir)

            pred = model.predict(X_test[:, ifeature].reshape(-1, 1))
            error = mean_absolute_error(pred, Y_test)
            
            feature_performance.append(error)

        perf[feature] = np.median(feature_performance)
        
    pickle.dump(dict(perf), open(f'{args.output_save}/feature_importance.p', 'wb'))

