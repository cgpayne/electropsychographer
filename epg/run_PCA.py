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
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

# external imports
import time
# import numpy as np
import pandas as pd

import epglib.params as p_epg
from config import user_config as cfg
import epglib.utils as ut
# import epglib.classes as cls


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- processing: {cfg.fname_pca}\n")
    
    # load in the data
    print("- loading in the data")
    X = pd.read_csv(cfg.inter_dir + '/' + cfg.fname_pca)
    print(X)
    
    # create response vector
    print("\n- generating response vector")
    demo = pd.read_csv(cfg.meta_dir + '/' + p_epg.fdemographic)
    all_subjects = {demo.loc[ii, 'subject']: demo.loc[ii, ' group'] for ii in range(len(demo))}
    print(f"  -- subjects (key) with respectives groups (value) = {all_subjects}")
    
    y = pd.DataFrame({'subject': list(X['subject']),
                      'class': [all_subjects[X.loc[ii, 'subject']] for ii in range(len(X))]})
    print("\n  -- y =")
    print(y)
    t_now = ut.time_stamp(t_now, t_zero, 'load + response')  # TIME STAMP
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
