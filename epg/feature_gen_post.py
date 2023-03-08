#!/usr/bin/env python
'''
feature_gen_post.py = do some post-processing on the results from feature_gen.py
  python3 feature_gen_post.py
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  this code performs post-processing on the results from feature_gen.py
  the roughly 55,230 features generated are reduced via PCA in the following script: run_PCA.py
  data is taken in from c_epg.fgen_dir, processed, and saved to c_epg.fgen_dir
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
import time
import math
import numpy as np
import pandas as pd
from tsfresh.utilities.dataframe_functions import impute

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut
# import epglib.classes as cls


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- post-processing: {cfg.fname_fgen_post_in}\n")
    
    # load in the data
    print("- loading in the data")
    df_extracted = pd.read_csv(c_epg.fgen_dir + '/' + cfg.fname_fgen_post_in, index_col='subject')
    
    # fill in empty values via imputation
    print("- imputing the data...")
    impute(df_extracted)
    print("  ...done.")
    t_now = ut.time_stamp(t_now, t_zero, 'imputation')  # TIME STAMP
    
    # # find all features with all zeros, and remove them
    # conditional = (df_extracted.T.iloc[:, 0] == 0)
    # for ii in range(1, len(df_extracted)):
    #     conditional = (df_extracted.T.iloc[:, ii] == 0) & conditional
    # df_allzeros = df_extracted.T[conditional]
    # num_zeros = len(df_allzeros)
    # num_features = df_extracted.shape[1]
    # print(f"- there are {num_zeros} ({100*num_zeros/num_features:.1f}%) out of {num_features} many generated features which are filled with all zeros\n")
    # print("  -- now removing said features")
    # df_extracted = df_extracted.drop(columns=df_allzeros.index)
    
    # find all features with mostly zeros and remove them
    print("- finding all features with mostly zeros")
    to_remove = []  # feature names to remove
    # count_rows = 0
    count_hits = 0  # count: number of rows with mostly zeros
    for feature, row in df_extracted.T.iterrows():
        # flattened: [0.0000001, -0.00000005, 0, 100, 20] -> [0, 0, 0, 100, 20]
        flattened = [0 if abs(row.iloc[jj]) < cfg.eps_flat else row.iloc[jj] for jj in range(len(row))]
        if np.count_nonzero(flattened) <= math.ceil(cfg.test_size_fgp*len(df_extracted)):
            # to_remove.append((count_rows, feature))
            to_remove.append(feature)
            count_hits += 1
        # count_rows += 1
    num_features = df_extracted.shape[1]
    print(f" -- there are {count_hits} ({100*count_hits/num_features:.1f}%) out of {num_features} many generated features which are filled with mostly zeros (per feature)")
    print("  -- now removing said features")
    df_extracted = df_extracted.drop(columns=to_remove)
    
    # output to csv
    print("- saving to file")
    ut.make_dir(c_epg.fgen_dir)
    df_extracted.to_csv(c_epg.fgen_dir + '/' + cfg.fname_fgen_post_out)
    t_now = ut.time_stamp(t_now, t_zero, 'check/remove all zeros, save')  # TIME STAMP
    
    
    print('\n\nF- I-- N---')
