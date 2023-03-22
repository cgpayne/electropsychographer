# Kfolds are split equally

# external imports
import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut
# from epglib.utils import eprint

cv_mode = sys.argv[1]  # 'kfold' = for K-fold CV, otherwise = for standard train_test_split


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
                train_test_split(self.X, self.y, test_size=cfg.test_size, random_state=math.floor(time.time()))
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=cfg.test_size, random_state=0)
        # dd = 74; X_train = X.drop(index=[dd]); X_test = X.loc[[dd]]; y_train = y.drop(index=[dd]); y_test = y.loc[[dd]]
    
    def _kfold_train_test_split(self, train: np.ndarray, test: np.ndarray):
        X_train = self.X.iloc[train, :].values
        X_test = self.X.iloc[test, :].values
        y_train = self.y[train].values
        y_test = self.y[test].values
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
            print(train, test)
            X_train, X_test, y_train, y_test = self._kfold_train_test_split(train, test)
            self.K_X_train.append(X_train)
            self.K_X_test.append(X_test)
            self.K_y_train.append(y_train)
            self.K_y_test.append(y_test)
    
    def perform_splitting(self):
        if self.cv_mode == 'kfold':
            self._kfold_split()
        else:
            self._standard_split()
    
    def save(self):
        print("- saving y's and X's to csv")
        if self.cv_mode == 'kfold':
            out_dir = c_epg.split_dir + '/' + cfg.split_data_handle + '_kfold-' + str(self.Kfolds)
            ut.make_dir(out_dir)
            for ii in range(self.Kfolds):
                print(f"  -- saving fold {ii}")
                self.K_y_train[ii].to_csv(out_dir + '/y_train-' + str(ii) + '_' + cfg.split_data_handle + '.csv')
                self.K_y_test[ii].to_csv(out_dir + '/y_test-' + str(ii) + '_' + cfg.split_data_handle + '.csv')
                np.savetxt(out_dir + '/X_train-' + str(ii) + '_' + cfg.split_data_handle + '.csv', self.K_X_train[ii], delimiter=',')
                np.savetxt(out_dir + '/X_test-' + str(ii) + '_' + cfg.split_data_handle + '.csv', self.K_X_test[ii], delimiter=',')
        else:
            out_dir = c_epg.split_dir + '/' + cfg.split_data_handle + '_standard'
            ut.make_dir(out_dir)
            self.y_train.to_csv(out_dir + '/y_train_' + cfg.split_data_handle + '.csv')
            self.y_test.to_csv(out_dir + '/y_test_' + cfg.split_data_handle + '.csv')
            np.savetxt(out_dir + '/X_train_' + cfg.split_data_handle + '.csv', self.X_train, delimiter=',')
            np.savetxt(out_dir + '/X_test_' + cfg.split_data_handle + '.csv', self.X_test, delimiter=',')


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- processing: {cfg.fname_split_in}")
    print(f"  pca_split_handle = {cfg.pca_split_handle}")
    if cv_mode == 'kfolds':
        print(f"  Kfolds = {cfg.Kfolds}")
    else:
        print(f"  test_size = {cfg.test_size}")
    print(f"  cv_mode = {cv_mode}\n")
    
    # load in the data
    print("- loading in the data")
    X = pd.read_csv(c_epg.fgen_dir + '/' + cfg.fname_split_in, index_col='subject')
    # X = X.iloc[:, 25807:25840]  # testing for bad features (eg, 25807)
    print(X)
    
    # create response vector
    print("\n- generating response vector")
    demo = pd.read_csv(c_epg.meta_dir + '/' + c_epg.fdemographic)
    all_subjects = {demo.loc[ii, 'subject']: demo.loc[ii, ' group'] for ii in range(len(demo))}
    print(f"  -- subjects (key) with respectives groups (value) = {all_subjects}")
    
    # in line below: take math.floor(X.index[ii]) to account for duplicated rows (eg, 13,13.1,13.2,...) in oversampling manually
    y = pd.Series([all_subjects[math.floor(X.index[ii])] for ii in range(len(X))], name='class')
    y.index = X.index
    print("\n  -- y =")
    print(y)
    t_now = ut.time_stamp(t_now, t_zero, 'load + response')  # TIME STAMP
    
    # perform the splitting and save
    ds = DataSplitter(X, y, cv_mode, cfg.rand_mode, cfg.Kfolds)
    ds.perform_splitting()
    t_now = ut.time_stamp(t_now, t_zero, 'splitting')  # TIME STAMP
    ds.save()
    t_now = ut.time_stamp(t_now, t_zero, 'save')  # TIME STAMP
    

    print('\n\nF- I-- N---')
