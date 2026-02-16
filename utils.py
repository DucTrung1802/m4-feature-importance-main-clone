import pandas as pd
import numpy as np
from pathlib import Path
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_error,
)
import os
import glob
import datetime as dt
from tqdm import tqdm
import re


def log(f):
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        res = f(*args, **kwargs)
        toc = dt.datetime.now()
        if hasattr(res, "shape"):
            print(f"{f.__name__} took={toc-tic} shape={res.shape}")
        else:
            print(f"{f.__name__} took={toc-tic}")
        return res

    return wrapper


def read_ts_from_files(files):
    ts_d = dict()
    for file in files:
        print(file)
        df = pd.read_csv(file)
        for index, row in df.iterrows():
            ts_name, *ts_values = row.tolist()
            ts_values = np.array(ts_values).astype(float)
            ts_values = ts_values[~np.isnan(ts_values)]
            ts_d[ts_name] = ts_values
    return ts_d


def differenced_dict(d):
    diff_d = {}
    for k, v in d.items():
        diff = pd.Series(v).diff().to_list()[1:]
        diff_d[k] = diff
    return diff_d


def log_dict(d):
    log_d = {}
    for k, v in d.items():
        log = list(np.log(v))
        log_d[k] = log
    return log_d


def get_tsfresh_features_df(ts_d):
    df = dict_to_df(ts_d)
    tf = tsfresh.extract_features(df, column_id="id")
    nunique = tf.nunique()
    cols_to_drop = nunique[nunique == 1].index
    tfm = tf.drop(cols_to_drop, axis=1).dropna(axis=1)
    return tfm


def create_directory_if_not_exist(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


@log
def get_problem_features(file):
    if isinstance(file, str):
        return pd.read_csv(file, index_col=0).sort_index()
    elif isinstance(file, list):
        dfs = [pd.read_csv(f, index_col=0) for f in file]
        return dfs[0].join(dfs[1:]).sort_index()
    else:
        raise Exception("File has to be a string or a list")


@log
def get_problem_algorithm_performance(file):
    return pd.read_csv(file, index_col=0).sort_index()


@log
def drop_nan_and_non_unique_columns(df):
    df = df.dropna(axis=1)
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


def re_find(patern, text):
    p = re.compile(patern)
    result = p.search(text)
    return result.group(1)


def feature_importance_to_feature_rank(df, feature_list):
    dfc = df.copy()
    ranked_df = dfc[feature_list].rank(axis=1)
    for c in feature_list:
        dfc[c] = ranked_df[c]
    return dfc


def forecasting_algo_type():
    return {
        "118": "Hybrid",
        "245": "Combination",
        "237": "Combination",
        "072": "Combination",
        "069": "Combination",
        "036": "Combination",
        "078": "Combination",
        "260": "Statistical",
        "238": "Combination",
        "039": "Combination",
        "005": "Statistical",
        "132": "Combination",
        "251": "Statistical",
        "250": "Combination",
        "243": "Combination",
        "235": "Statistical",
        "104": "Combination",
        "Theta": "Statistical",
        "Com": "Combination",
        "ARIMA": "Statistical",
        "223": "Combination",
        "Damped": "Statistical",
        "ETS": "Statistical",
        "239": "Combination",
        "211": "Machine Learning",
        "231": "Combination",
        "227": "Combination",
        "082": "Statistical",
        "212": "Combination",
        "236": "Combination",
        "248": "Combination",
        "030": "Statistical",
        "Holt": "Statistical",
        "SES": "Statistical",
        "234": "Combination",
        "024": "Statistical",
        "Naive2": "Statistical",
        "218": "Statistical",
        "106": "Statistical",
        "043": "Combination",
        "Naive": "Statistical",
        "216": "Combination",
        "sNaive": "Statistical",
        "169": "Combination",
        "241": "Combination",
        "191": "Combination",
        "126": "Combination",
        "244": "Machine Learning",
        "070": "Combination",
        "249": "Statistical",
        "252": "Statistical",
        "255": "Statistical",
        "009": "Statistical",
        "256": "Statistical",
        "253": "Statistical",
        "091": "Machine Learning",
        "RNN": "Machine Learning",
        "219": "Machine Learning",
        "MLP": "Machine Learning",
        "225": "Statistical",
        "258": "Statistical",
    }


def get_meta_model_name_map():
    return {
        "M4KerasNetRegressionModel": "Neural network",
        "M4DummyMeanRegression": "Mean",
        "M4KNeighborsRegressor": "KNN [Euclidean]",
        "M4RandomForestRegressor": "Random forest [MT]",
        "M4KNeighborsCosineRegressor": "KNN [Cosine]",
        "M4XGBRegressor": "XGBoost",
        "M4RandomForestSingleOutputRegressor": "Random forest [ST]",
    }


def get_correlation_map():
    return {
        "0.6": "0.60",
        "0.95": "0.95",
        "0.8": "0.80",
        "0.9": "0.90",
        "0.7": "0.70",
        "1.0": "1.00",
        "0.5": "0.50",
    }


def get_feature_set_map():
    return {
        "TSFresh,TSFreshDiff,TSFreshLog": "tsfresh [raw, diff, log]",
        "Catch22": "catch22 [raw]",
        "TSFresh,Catch22": "catch22 [raw], tsfresh [raw]",
        "Catch22,Catch22Diff,Catch22Log": "catch22 [raw, diff, log]",
        "TSFresh": "tsfresh [raw]",
    }


def get_feature_name_map():
    return {
        "value__has_duplicate_max": "has_duplicate_max",
        "value__mean_second_derivative_central": "mean_second_derivative_central",
        "value__median": "median",
        "value__length": "length",
        "value__standard_deviation": "standard_deviation",
        "value__skewness": "skewness",
        "value__first_location_of_maximum": "first_location_of_maximum",
        "value__first_location_of_minimum": "first_location_of_minimum",
        "value__percentage_of_reoccurring_values_to_all_values": "percentage_of_reoccurring_values_to_all_values",
        "value__sum_of_reoccurring_data_points": "sum_of_reoccurring_data_points",
        "value__ratio_value_number_to_time_series_length": "ratio_value_number_to_time_series_length",
        "value__benford_correlation": "benford_correlation",
        "value__symmetry_looking__r_0.05": "symmetry_looking_r_0.05",
        "value__symmetry_looking__r_0.1": "symmetry_looking_r_0.1",
        "value__symmetry_looking__r_0.15000000000000002": "symmetry_looking_r_0.15",
        "value__symmetry_looking__r_0.2": "symmetry_looking_r_0.2",
        "value__symmetry_looking__r_0.30000000000000004": "symmetry_looking_r_0.30",
        "value__symmetry_looking__r_0.4": "symmetry_looking_r_0.4",
        "value__symmetry_looking__r_0.45": "symmetry_looking_r_0.45",
        "value__large_standard_deviation__r_0.05": "large_standard_deviation_r_0.05",
        "value__large_standard_deviation__r_0.1": "large_standard_deviation_r_0.1",
        "value__large_standard_deviation__r_0.15000000000000002": "large_standard_deviation_r_0.15",
        "value__large_standard_deviation__r_0.2": "large_standard_deviation_r_0.2",
        "value__large_standard_deviation__r_0.30000000000000004": "large_standard_deviation_r_0.30",
        "value__large_standard_deviation__r_0.35000000000000003": "large_standard_deviation_r_0.35",
        "value__large_standard_deviation__r_0.4": "large_standard_deviation_r_0.4",
        "value__large_standard_deviation__r_0.45": "large_standard_deviation_r_0.45",
        "value__autocorrelation__lag_0": "autocorrelation_lag_0",
        "value__autocorrelation__lag_2": "autocorrelation_lag_2",
        "value__autocorrelation__lag_9": "autocorrelation_lag_9",
        "value__partial_autocorrelation__lag_2": "partial_autocorrelation_lag_2",
        "value__partial_autocorrelation__lag_3": "partial_autocorrelation_lag_3",
        "value__partial_autocorrelation__lag_4": "partial_autocorrelation_lag_4",
        "value__partial_autocorrelation__lag_5": "partial_autocorrelation_lag_5",
        "value__cwt_coefficients__coeff_0__w_5__widths_(2, 5, 10, 20)": "cwt_coefficients_coeff_0_w_5_widths_(2, 5, 10, 20)",
        "value__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)": "cwt_coefficients_coeff_0_w_20_widths_(2, 5, 10, 20)",
        "value__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)": "cwt_coefficients_coeff_8_w_2_widths_(2, 5, 10, 20)",
        "value__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)": "cwt_coefficients_coeff_10_w_2_widths_(2, 5, 10, 20)",
        "value__spkt_welch_density__coeff_5": "spkt_welch_density_coeff_5",
        "value__ar_coefficient__coeff_10__k_10": "ar_coefficient_coeff_10_k_10",
        'value__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8': 'change_quantiles_f_agg_"mean"_isabs_False_qh_1.0_ql_0.8',
        'value__fft_coefficient__attr_"real"__coeff_1': 'fft_coefficient_attr_"real"_coeff_1',
        'value__fft_coefficient__attr_"real"__coeff_2': 'fft_coefficient_attr_"real"_coeff_2',
        'value__fft_coefficient__attr_"real"__coeff_4': 'fft_coefficient_attr_"real"_coeff_4',
        'value__fft_coefficient__attr_"real"__coeff_6': 'fft_coefficient_attr_"real"_coeff_6',
        'value__fft_coefficient__attr_"imag"__coeff_6': 'fft_coefficient_attr_"imag"_coeff_6',
        'value__linear_trend__attr_"pvalue"': 'linear_trend_attr_"pvalue"',
        'value__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"': 'agg_linear_trend_attr_"slope"_chunk_len_10_f_agg_"var"',
        'value__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"': 'agg_linear_trend_attr_"stderr"_chunk_len_5_f_agg_"max"',
        'value__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"': 'agg_linear_trend_attr_"stderr"_chunk_len_10_f_agg_"mean"',
        'value__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"': 'augmented_dickey_fuller_attr_"teststat"_autolag_"AIC"',
        'value__augmented_dickey_fuller__attr_"usedlag"__autolag_"AIC"': 'augmented_dickey_fuller_attr_"usedlag"_autolag_"AIC"',
        "value__energy_ratio_by_chunks__num_segments_10__segment_focus_4": "energy_ratio_by_chunks_num_segments_10_segment_focus_4",
        "value__energy_ratio_by_chunks__num_segments_10__segment_focus_5": "energy_ratio_by_chunks_num_segments_10_segment_focus_5",
        "value__energy_ratio_by_chunks__num_segments_10__segment_focus_6": "energy_ratio_by_chunks_num_segments_10_segment_focus_6",
        "value__ratio_beyond_r_sigma__r_0.5": "ratio_beyond_r_sigma_r_0.5",
        "value__ratio_beyond_r_sigma__r_1": "ratio_beyond_r_sigma_r_1",
        "value__ratio_beyond_r_sigma__r_1.5": "ratio_beyond_r_sigma_r_1.5",
        "value__ratio_beyond_r_sigma__r_2.5": "ratio_beyond_r_sigma_r_2.5",
        "value__ratio_beyond_r_sigma__r_3": "ratio_beyond_r_sigma_r_3",
        "value__ratio_beyond_r_sigma__r_6": "ratio_beyond_r_sigma_r_6",
        "value__lempel_ziv_complexity__bins_2": "lempel_ziv_complexity_bins_2",
        "value__permutation_entropy__dimension_3__tau_1": "permutation_entropy_dimension_3_tau_1",
    }


def get_meta_importance_name_pair_map():
    return {
        "M4DummyMeanRegression,permutation": "P/Mean",
        "M4KerasNetRegressionModel,permutation": "P/NN",
        "M4RandomForestRegressor,permutation": "P/RF",
        "M4RandomForestSingleOutputRegressor,permutation": "P/SRF",
        "M4RandomForestSingleOutputRegressor,shap": "S/SRF",
        "M4XGBRegressor,permutation": "P/XGB",
        "M4RandomForestSingleOutputRegressor,randomforest": "RF/SRF",
        "M4DummyMeanRegression,shap": "S/Mean",
        "M4KerasNetRegressionModel,shap": "S/NN",
        "M4RandomForestRegressor,shap": "S/RF",
        "M4XGBRegressor,shap": "S/XGB",
        "M4XGBRegressor,xgboost-cover": "C/XGB",
        "M4XGBRegressor,xgboost-gain": "G/XGB",
        "M4XGBRegressor,xgboost-total_cover": "TC/XGB",
        "M4XGBRegressor,xgboost-total_gain": "TG/XGB",
        "M4XGBRegressor,xgboost-weight": "W/XGB",
    }
