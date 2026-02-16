from abc import ABC, abstractmethod
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from utils import *
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from sklearn.neighbors import LSHForest
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


class M4Model():
    _registry = {}

    def __init_subclass__(cls, is_registered=True, **kwargs):
        super().__init_subclass__(**kwargs)
        if is_registered:
            M4Model._registry[cls.__name__] = cls

    @staticmethod
    def get_registered_models():
        return M4Model._registry

    @staticmethod
    def get_registered_model(name):
        return M4Model._registry[name]
    
    @abstractmethod
    def fit(self, X_train, Y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def set_single_predict(self, column):
        pass

    
class SklearnM4Model(M4Model, ABC, is_registered=False):
    def __init__(self, model):
        self.model = model
        self.predict_one_column=None
        
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
    
    def predict(self, X_test):
        pred = self.model.predict(X_test)
        
        if hasattr(self, 'predict_one_column') and self.predict_one_column is not None:
            return pred[:, self.predict_one_column]
        return pred
    
    def save(self, directory):
        create_directory_if_not_exist(directory)
        pickle.dump(self, open(directory + '/model.p', "wb"))
        
    @staticmethod
    def load_model(directory):
        return pickle.load(open(directory + '/model.p', "rb"))
    
    def set_single_predict(self, column):
        self.predict_one_column=column
    
    
class M4MLPRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = MLPRegressor(random_state=self.seed, max_iter=1000, learning_rate_init=0.0001)
        super().__init__(model)
    
class M4MLPRegressorLarge(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = MLPRegressor(random_state=self.seed, hidden_layer_sizes=(200, 200), max_iter=500, learning_rate_init=0.0005)
        super().__init__(model)
        
class M4MLPRegressorSmall(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = MLPRegressor(random_state=self.seed, hidden_layer_sizes=(50), max_iter=500, learning_rate_init=0.0005)
        super().__init__(model)
        
class M4DecisionTreeRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = DecisionTreeRegressor(random_state=self.seed, criterion='squared_error')
        super().__init__(model)
        
class M4RandomForestRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = RandomForestRegressor(random_state=self.seed, criterion='squared_error', verbose=verbose, n_jobs=-1)
        super().__init__(model)
        
class M4RandomForestSingleOutputRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        self.seed = seed
        model = MultiOutputRegressor(RandomForestRegressor(random_state=self.seed, criterion='squared_error', verbose=verbose, n_estimators=50, min_samples_split=4, min_samples_leaf=2), n_jobs=-1)
        super().__init__(model)
        
    def predict(self, X_test):
        if hasattr(self, 'predict_one_column') and self.predict_one_column is not None:
            return self.model.estimators_[self.predict_one_column].predict(X_test)
        return self.model.predict(X_test)
    
class M4KNeighborsRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        model = KNeighborsRegressor(n_jobs=-1)
        super().__init__(model)
        
class M4KNeighborsCosineRegressor(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        model = KNeighborsRegressor(metric='cosine', n_jobs=-1)
        super().__init__(model)
        
class M4DummyMeanRegression(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        model = DummyRegressor(strategy="mean")
        super().__init__(model)
        
class M4DummyMedianRegression(SklearnM4Model):
    def __init__(self, seed=0, verbose=0):
        model = DummyRegressor(strategy="median")
        super().__init__(model)     
    
class M4XGBRegressor(M4Model):
    def load_model(directory):
        return pickle.load(open(directory + '/model.p', "rb"))
    
    def __init__(self, seed=0):
        self.seed = seed
        self.model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'), n_jobs=-1)
        
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
    
    def predict(self, X_test):
        #pred = self.model.predict(X_test)
        if hasattr(self, 'predict_one_column') and self.predict_one_column is not None:
            #return pred[:, self.predict_one_column]
            return self.model.estimators_[self.predict_one_column].predict(X_test)
        #return pred
        return self.model.predict(X_test)
    
    def save(self, directory):
        create_directory_if_not_exist(directory)
        pickle.dump(self, open(directory + '/model.p', "wb"))
        
    def set_single_predict(self, column):
        self.predict_one_column=column
    
    
class M4KerasNetRegressionModel(M4Model):
    @staticmethod
    def load_model(directory):
        model = M4KerasNetRegressionModel()
        from keras.models import load_model
        model.model = load_model(f'{directory}/model.h5')
        return model

    def __init__(self, seed=0, verbose=0):
        self.model = None
        self.num_hidden_layers = 3
        self.hidden_layer_size = 500
        self.seed=seed
        self.verbose=verbose

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        if hasattr(self, 'predict_one_column') and self.predict_one_column is not None:
            return pred[:, self.predict_one_column]
        return pred

    def fit(self, X, Y):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_size, input_shape=(X.shape[1],), activation="relu"))
        for li in range(self.num_hidden_layers):
            self.model.add(Dense(self.hidden_layer_size, activation="relu"))
        self.model.add(Dense(Y.shape[1], activation="sigmoid"))
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)
        
        adam = keras.optimizers.Adam(learning_rate=0.0001)
        
        self.model.compile(loss='mse', optimizer=adam)
        self.history_ = self.model.fit(X_train, Y_train, epochs=70, batch_size=256, verbose=self.verbose, validation_data=(X_val, Y_val), callbacks=[callback])

    def save(self, directory):
        create_directory_if_not_exist(directory)
        self.model.save(f'{directory}/model.h5')
        
    def set_single_predict(self, column):
        self.predict_one_column=column
    
    
    
    
    
    
    
    
    
    
    
    
    
