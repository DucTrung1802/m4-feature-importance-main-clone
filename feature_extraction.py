import argparse
import glob
from utils import read_ts_from_files, create_directory_if_not_exist
from feature_extractors import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=list(FeatureExtractor.get_registered_models().keys()))
    args = parser.parse_args()
    
    print(f'Feature extraction method: {args.method}')
    
    files = glob.glob(args.input_folder)
    
    timeseries_dictionary = read_ts_from_files(files)
    
    ex = FeatureExtractor.get_registered_model(args.method)()
    df = ex.transform(timeseries_dictionary)
    
    create_directory_if_not_exist(os.path.dirname(args.output))
    df.to_csv(args.output)
