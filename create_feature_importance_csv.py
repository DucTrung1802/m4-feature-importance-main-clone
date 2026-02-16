import pickle
import glob
from utils import *
import pandas as pd

fdf = get_problem_features(glob.glob('features/TSFresh.csv'))
fdf = fdf.pipe(drop_nan_and_non_unique_columns)
all_columns = list(fdf.columns)

files = glob.glob('feature_importance/*.p')

df_dict = {x: [] for x in ['ml_algorithm', 'forecasting_algorithm', 'run', 'feature_importance_method'] + all_columns}

for file in files:
    algo = re_find('m_(.+?)_f_', file)
    run = re_find('r_(.+?)_m_', file)
    fe = re_find('_f_(.+?)\.p', file)
    
    meta = pickle.load(open(file, 'rb'))
    
    if 'shap' in fe:
        fe = 'shap'
        
    for forecasting_algo, perf_list in meta.items():
        df_dict['ml_algorithm'].append(algo)
        df_dict['forecasting_algorithm'].append(forecasting_algo)
        df_dict['run'].append(run)
        df_dict['feature_importance_method'].append(fe)
        
        perf_dict = {b:a for a, b in perf_list}
        
        for feature in all_columns:
            if feature in perf_dict:
                df_dict[feature].append(perf_dict[feature])
            else:
                df_dict[feature].append(None)
                
dffe = pd.DataFrame(df_dict)
dffe.to_csv('feature_importance/merged_feature_importance.csv')

check_if_complete_df = dffe.groupby(['ml_algorithm', 'forecasting_algorithm', 'feature_importance_method']).count().reset_index()

if len(check_if_complete_df.query("run!=30")) != 0:
    print("Not all feature values are avaiable")
