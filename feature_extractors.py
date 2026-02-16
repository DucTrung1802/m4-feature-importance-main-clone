from abc import ABC
from abc import abstractmethod
from tqdm import tqdm
import catch22
import pandas as pd
import tsfresh
import numpy as np

class FeatureExtractor(ABC):
    _registry = {}

    def __init_subclass__(cls, is_registered=True, **kwargs):
        super().__init_subclass__(**kwargs)
        if is_registered:
            FeatureExtractor._registry[cls.__name__] = cls

    @staticmethod
    def get_registered_models():
        return FeatureExtractor._registry

    @staticmethod
    def get_registered_model(name):
        return FeatureExtractor._registry[name]

    @abstractmethod
    def transform(self, time_series_dict):
        pass
    
    def differenced_dict(self, d):
        diff_d = {}
        for k, v in d.items():
            diff = pd.Series(v).diff().to_list()[1:]
            diff_d[k] = diff
        return diff_d

    def log_dict(self, d):
        log_d = {}
        for k, v in d.items():
            log = list(np.log(v))
            log_d[k] = log
        return log_d


class Catch22(FeatureExtractor):
    def extract_features(self, time_series_dict):
        data = {'index': []}
        for k, v in tqdm(list(time_series_dict.items())):
            catchOut = catch22.catch22_all(v)
            
            data['index'].append(k)
            
            for n, v in zip(catchOut['names'],catchOut['values']):
                if n not in data.keys():
                    data[n] = []
                data[n].append(v)

        df = pd.DataFrame(data)
        df = df.set_index('index')
        df.index.name = None
        return df
    
    def transform(self, time_series_dict):
        return self.extract_features(time_series_dict)
    
class Catch22Diff(Catch22):
    def transform(self, time_series_dict):
        diff_time_series_dict = self.differenced_dict(time_series_dict)
        tdf = self.extract_features(diff_time_series_dict)
        tdf = tdf.rename(columns={x: f'diff_{x}' for x in tdf.columns})
        return tdf
    
class Catch22Log(Catch22):
    def transform(self, time_series_dict):
        log_time_series_dict = self.log_dict(time_series_dict)
        tdf = self.extract_features(log_time_series_dict)
        tdf = tdf.rename(columns={x: f'log_{x}' for x in tdf.columns})
        return tdf
    
class TSFreshFeatureAbstractExtractor(FeatureExtractor, is_registered=False):
    def dict_to_df(self, d):
        ids = []
        values = []
        for k, v in d.items():
            for vn in v:
                ids.append(k)
                values.append(vn)
        data = {'id': ids, 'value':values}
        return pd.DataFrame(data)

class TSFresh(TSFreshFeatureAbstractExtractor):   
    def transform(self, time_series_dict):
        df = self.dict_to_df(time_series_dict)
        tf=tsfresh.extract_features(df, column_id='id')
        return tf
    
class TSFreshDiff(TSFreshFeatureAbstractExtractor):   
    def transform(self, time_series_dict):
        diff_time_series_dict = self.differenced_dict(time_series_dict)
        df = self.dict_to_df(diff_time_series_dict)
        tf=tsfresh.extract_features(df, column_id='id')
        tf = tf.rename(columns={x: f'diff_{x}' for x in tf.columns})
        return tf
    
class TSFreshLog(TSFreshFeatureAbstractExtractor):   
    def transform(self, time_series_dict):
        log_time_series_dict = self.log_dict(time_series_dict)
        df = self.dict_to_df(log_time_series_dict)
        tf=tsfresh.extract_features(df, column_id='id')
        tf = tf.rename(columns={x: f'log_{x}' for x in tf.columns})
        return tf

