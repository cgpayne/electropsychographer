#!/usr/bin/env python
'''
split_the_data.py = split the data out into training and test data, with K-fold option
  python3 split_the_data.py <cv_mode>
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  this code will split the data into training and test sets, depending on the cv_mode option
  if cv_mode = 'kfold', then we use StratifiedKFold() and split out train/test according to the folds
    otherwise, we simply use train_test_split() as a "standard" splitting
  note that each k-fold is split into equal sizes, ie, 5-fold gives 20%, 20%, 20%, 20%, 20% from random slots
    for a splitting, one of those folds is used as a test and the rest combines into the training, and so on
  data is taken in from c_epg.fgen_dir, processed, and saved to c_epg.split_dir
NOTES
  [none]
RESOURCES
  [none]
CONVENTIONS
  [none]
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

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
    '''
    CLASS: DataSplitter = splits the data accordingly (K-fold or otherwise)
    '''
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_mode: str, rand_mode: str, Kfolds: int):
        self.X = X
        self.y = y
        self.cv_mode = cv_mode
        self.rand_mode = rand_mode
        self.Kfolds = Kfolds
    
    def _standard_split(self):
        '''
        METHOD: _standard_split = use train_test_split() to split the data via random seed or random_state=0
        '''
        if self.rand_mode == 'random':
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=cfg.test_size, random_state=math.floor(time.time()))
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=cfg.test_size, random_state=0)
        # dd = 74; X_train = X.drop(index=[dd]); X_test = X.loc[[dd]]; y_train = y.drop(index=[dd]); y_test = y.loc[[dd]]
    
    def _kfold_train_test_split(self, train: np.ndarray, test: np.ndarray):
        '''
        METHOD: _kfold_train_test_split = this acts as the K-fold'ed version of train_test_split()
        IN: train = a list of indices designating the train data (the rest of the folds)
            test = a list of indices designating the test data (a single fold)
        OUT: X_train, X_test, y_train, y_test = the actual data split by the folding
        '''
        # NOTE: StratifiedKFold() splits in terms of iloc indices
        X_train = self.X.iloc[train, :].values
        X_test = self.X.iloc[test, :].values
        y_train = self.y.iloc[train]
        y_test = self.y.iloc[test]
        return X_train, X_test, y_train, y_test
    
    def _kfold_split(self):
        '''
        METHOD: _kfold_split = this is the K-fold'ed version of _standard_split above
                               it uses StratifiedKFold() and _kfold_train_test_split() via random seed or otherwise
        '''
        if self.rand_mode == 'random':
            skf = StratifiedKFold(n_splits=self.Kfolds, shuffle=True, random_state=math.floor(time.time()))
        else:
            skf = StratifiedKFold(n_splits=self.Kfolds, shuffle=True, random_state=0)
        self.K_X_train = []
        self.K_X_test = []
        self.K_y_train = []
        self.K_y_test = []
        for train, test in skf.split(self.X, self.y):
            print(train, test)
            X_train, X_test, y_train, y_test = self._kfold_train_test_split(train, test)
            self.K_X_train.append(X_train)
            self.K_X_test.append(X_test)
            self.K_y_train.append(y_train)
            self.K_y_test.append(y_test)
    
    def perform_splitting(self):
        '''
        METHOD: perform_splitting = run _kfold_split() or _standard_split()
        '''
        if self.cv_mode == 'kfold':
            self._kfold_split()
        else:
            self._standard_split()
    
    def save(self):
        '''
        METHOD: save = save all the split out data into their respective csv's
        '''
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
    print(f"  split_data_handle = {cfg.split_data_handle}")
    if cv_mode == 'kfold':
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
