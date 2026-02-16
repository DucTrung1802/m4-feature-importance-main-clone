import argparse
import glob
from feature_extractors import * 
from utils import read_ts_from_files, create_directory_if_not_exist
import os
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_absolute_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--input_predictions', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    files = glob.glob(args.input_data)
    data = read_ts_from_files(files)

    keys = list(data.keys())
    performance_dict = {'Problem': keys}

    for file in glob.glob(args.input_predictions):
        algorithm = os.path.basename(file).replace('submission-', '').replace('.csv', '')
        data_algo = read_ts_from_files([file])

        errors = []

        for key in keys:
            error = mean_absolute_percentage_error(data_algo[key], data[key])
            errors.append(error)
        performance_dict[algorithm] = errors
        
    pdf = pd.DataFrame(performance_dict).set_index('Problem')
    
    create_directory_if_not_exist(os.path.dirname(args.output))
    pdf.to_csv(args.output)
