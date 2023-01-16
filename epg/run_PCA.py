#!/usr/bin/env python
#  run_PCA.py = run the Principal Component Analysis (PCA)
#  python3 run_PCA.py
#  Copyright (c) 2022 Charlie Payne
#  Licence: GNU GPLv3
# DESCRIPTION
#  [insert: here]
# NOTES
#  [none]
# RESOURCES
#  -- https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
#  -- https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
#  -- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

# external imports
import time
import math
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# internal imports
import epglib.params as p_epg
from config import user_config as cfg
import epglib.utils as ut
# import epglib.classes as cls


if __name__ == '__main__':
    ## set up for the PCA
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- processing: {cfg.fname_pca}\n")
    
    # load in the data
    print("- loading in the data")
    X = pd.read_csv(cfg.inter_dir + '/' + cfg.fname_pca, index_col='subject')
    # X.index = X['subject']
    # X = X.drop(columns=['subject'])
    print(X)
    
    # create response vector
    print("\n- generating response vector")
    demo = pd.read_csv(cfg.meta_dir + '/' + p_epg.fdemographic)
    all_subjects = {demo.loc[ii, 'subject']: demo.loc[ii, ' group'] for ii in range(len(demo))}
    print(f"  -- subjects (key) with respectives groups (value) = {all_subjects}")
    
    y = pd.Series([all_subjects[X.index[ii]] for ii in range(len(X))], name='class')
    y.index = X.index
    # y = pd.DataFrame({'subject': list(X['subject']),
    #                   'class': [all_subjects[X.loc[ii, 'subject']] for ii in range(len(X))]})
    print("\n  -- y =")
    print(y)
    t_now = ut.time_stamp(t_now, t_zero, 'load + response')  # TIME STAMP
    
    # split the data: training, testing
    print("- split the dataset into training data and test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=0)
    # _train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=math.floor(t_now))
    
    print("\n  -- X_train =")
    print(X_train)
    print("\n  -- y_train =")
    print(y_train)
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("\n  -- X_test =")
    print(X_test)
    print("\n  -- y_test =")
    print(y_test)
    t_now = ut.time_stamp(t_now, t_zero, 'train/test split')  # TIME STAMP
    
    # scale the data for optimized performance
    print('- scale the data via Z-score')
    sc = StandardScaler()  # scale via Z-score
    X_train = sc.fit_transform(X_train)  # fit and transform the X_train data via Z-score
    X_test = sc.transform(X_test)        # transform the X_test data using the mean and standard deviation fit from X_train
    
    print("\n  -- X_train =")
    print(X_train)
    print("\n  -- X_test =")
    print(X_test)
    t_now = ut.time_stamp(t_now, t_zero, 'scaled data')  # TIME STAMP
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
