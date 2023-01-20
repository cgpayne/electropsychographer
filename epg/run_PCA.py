#!/usr/bin/env python
'''
run_PCA.py = run the Principal Component Analysis (PCA)
  python3 run_PCA.py
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  this takes in the output from feature_gen.py and runs a PCA on it
  thus reducing the 55k generated features from before to on the order of 10
  data is taken in from c_epg.inter_dir and saved to c_epg.inter_dir
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
import pandas as pd

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut
import epglib.classes as cls


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
    
    ## build the model
    epg = cls.DataEPG(X, y)
    epg.print_Xy()
    t_now = ut.time_stamp(t_now, t_zero, 'train/test split')  # TIME STAMP
    
    # scale the data for optimized performance
    epg.scale_X()
    epg.print_X()
    t_now = ut.time_stamp(t_now, t_zero, 'scaled data')  # TIME STAMP
    
    ## perform the PCA
    epg.exec_PCA()
    epg.print_X()
    t_now = ut.time_stamp(t_now, t_zero, 'PCA')  # TIME STAMP
    
    # plot the explained variance
    epg.set_ev()
    print("  -- plotting...")
    fig_dir_now = c_epg.fig_dir + '/' + cfg.data_handle
    ut.make_dir(fig_dir_now)
    epg.plot_ev(fig_dir_now)
    
    # plot the cummulative explained variance
    epg.plot_cev(fig_dir_now)
    
    # split out training data into control (HC) and schizophrenia (SZ)
    epg.set_HCSZ()
    
    # plot PC1 vs PC2 then PC2 vs PC3
    epg.plot_PC(fig_dir_now)
    
    print("     ...done.")
    t_now = ut.time_stamp(t_now, t_zero, 'plotting')  # TIME STAMP
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
