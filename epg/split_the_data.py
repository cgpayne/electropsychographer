# external imports
# import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut
from epglib.utils import eprint

cv_mode = sys.argv[1]  # 'kfold' for K-fold CV, otherwise for standard train_test_split


class DataSplitter():
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_mode: str, rand_mode: str, Kfolds: int):
        self.X = X
        self.y = y
        self.cv_mode = cv_mode
        self.rand_mode = rand_mode
        self.Kfolds = Kfolds
    
    def _standard_split(self):
        if self.rand_mode == 'random':
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=cfg.test_size, random_state=math.floor(time.time()))
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=cfg.test_size, random_state=0)
        # dd = 74; X_train = X.drop(index=[dd]); X_test = X.loc[[dd]]; y_train = y.drop(index=[dd]); y_test = y.loc[[dd]]
    
    def _kfold_train_test_split(self, train: np.ndarray, test: np.ndarray):
        X_train = self.X.iloc[train, :]
        X_test = self.X.iloc[test, :]
        y_train = self.y[train]
        y_test = self.y[test]
        return X_train, X_test, y_train, y_test
    
    def _kfold_split(self):
        if self.rand_mode == 'random':
            skf = StratifiedKFold(n_splits=self.Kfolds, random_state=math.floor(time.time()))
        else:
            skf = StratifiedKFold(n_splits=self.Kfolds, random_state=0)
        self.K_X_train = []
        self.K_X_test = []
        self.K_train = []
        self.K_test = []
        for train, test in skf.split(self.X, self.y):
            X_train, X_test, y_train, y_test = self._kfold_train_test_split(self, train, test)
            self.K_X_train.append(X_train)
            self.K_X_test.append(X_test)
            self.K_y_train.append(y_train)
            self.K_y_test.append(y_test)
    
    def perform_splitting(self):
        if self.cv_mode == 'kfold':
            self._kfold_split(self)
        else:
            self._standard_split(self)
    
    def save(self):
        if self.cv_mode == 'kfold':
            asdf
        else:
            # TODO need to make file names in config, etc
