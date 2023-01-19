#!/usr/bin/env python
'''
run_PCA.py = run the Principal Component Analysis (PCA)
  python3 run_PCA.py
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  [insert: here]
NOTES
  [none]
RESOURCES
  -- https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
  -- https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
  -- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
CONVENTIONS
  [none]
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

# external imports
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# internal imports
import epglib.constants as c_epg
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
    X = pd.read_csv(c_epg.inter_dir + '/' + cfg.fname_pca, index_col='subject')
    # X.index = X['subject']
    # X = X.drop(columns=['subject'])
    print(X)
    
    # create response vector
    print("\n- generating response vector")
    demo = pd.read_csv(c_epg.meta_dir + '/' + c_epg.fdemographic)
    all_subjects = {demo.loc[ii, 'subject']: demo.loc[ii, ' group'] for ii in range(len(demo))}
    print(f"  -- subjects (key) with respectives groups (value) = {all_subjects}")
    
    y = pd.Series([all_subjects[X.index[ii]] for ii in range(len(X))], name='class')
    y.index = X.index
    print("\n  -- y =")
    print(y)
    t_now = ut.time_stamp(t_now, t_zero, 'load + response')  # TIME STAMP
    
    # split the data: training, testing
    print("- split the dataset into training data and test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=math.floor(t_now))
    
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
    
    ## perform the PCA
    print("\n- perform the PCA")
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print("\n  -- X_train =")
    print(X_train)
    print("\n  -- X_test =")
    print(X_test)
    t_now = ut.time_stamp(t_now, t_zero, 'PCA')  # TIME STAMP
    
    # plot the explained variance
    explained_variance = pca.explained_variance_ratio_
    num_components = len(explained_variance)
    print(f"  -- explained_variance = {explained_variance}")
    print(f"  -- number of components = {num_components}")
    
    print("  -- plotting...")
    fig_dir_now = c_epg.fig_dir + '/' + cfg.data_handle
    ut.make_dir(fig_dir_now)
    
    print("    --- ...explained variance...")
    fig_ev = plt.figure(1)
    plt.bar(range(1, num_components+1), explained_variance)
    plt.title('explained variance')
    plt.xlabel('component')
    plt.ylabel('percent explained')
    plt.xticks([ii for ii in range(1, num_components+1)])
    plt.savefig(fig_dir_now + '/explained_variance.pdf')
    if cfg.pca_show_fig == 'on':
        plt.show()
    
    # plot the cummulative explained variance
    sum_cev = 0
    cummulative_ev = []
    for ii in range(num_components):
        sum_cev += explained_variance[ii]
        cummulative_ev.append(sum_cev)
    print(cummulative_ev)
    
    print("    --- ...cummulative explained variance...")
    fig_cev = plt.figure(2)
    plt.bar(range(1, num_components+1), cummulative_ev)
    plt.title('cummulative explained variance')
    plt.xlabel('component')
    plt.ylabel('percent explained')
    plt.xticks([ii for ii in range(1, num_components+1)])
    plt.savefig(fig_dir_now + '/cummulative_explained_variance.pdf')
    if cfg.pca_show_fig == 'on':
        plt.show()
    
    # split out training data into control (HC) and schizophrenia (SZ)
    X_HC = np.empty((0, X_train.shape[1]))
    X_SZ = np.empty((0, X_train.shape[1]))

    for ii in range(len(y_train)):
        if y_train.iloc[ii] == 0:
            X_HC = np.append(X_HC, [X_train[ii, :]], axis=0)
        elif y_train.iloc[ii] == 1:
            X_SZ = np.append(X_SZ, [X_train[ii, :]], axis=0)
    
    # plot PC1 vs PC2 then PC2 vs PC3
    print("    --- ...PC1 vs PC2...")
    fig12 = plt.figure(3)
    plt.scatter(X_HC[:, 0], X_HC[:, 1], label='HC')
    plt.scatter(X_SZ[:, 0], X_SZ[:, 1], label='SZ')
    plt.legend()
    plt.title('PC1 vs PC2')
    plt.xlabel(f"PC1 = {100*explained_variance[0]:.2f}%")
    plt.ylabel(f"PC2 = {100*explained_variance[1]:.2f}%")
    plt.savefig(fig_dir_now + '/PC1_vs_PC2.pdf')
    if cfg.pca_show_fig == 'on':
        plt.show()
    
    print("    --- ...PC2 vs PC3...")
    fig23 = plt.figure(4)
    plt.scatter(X_HC[:, 1], X_HC[:, 2], label='HC')
    plt.scatter(X_SZ[:, 1], X_SZ[:, 2], label='SZ')
    plt.title('PC2 vs PC3')
    plt.xlabel(f"PC2 = {100*explained_variance[1]:.2f}%")
    plt.ylabel(f"PC3 = {100*explained_variance[2]:.2f}%")
    plt.legend()
    plt.savefig(fig_dir_now + '/PC2_vs_PC3.pdf')
    if cfg.pca_show_fig == 'on':
        plt.show()
    
    print("     ...done.")
    t_now = ut.time_stamp(t_now, t_zero, 'plotting')  # TIME STAMP
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
