#!/usr/bin/env python
'''
user_config.py = holds user configuration parameters for the electropsychographer
  from config import user_config as cfg
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  user_config.py is the main configuration file
  to set this up: copy example_config.py -> user_config.py
  example_config.py will be tracked by git, but will *not* be imported
  user_config.py will *not* be tracked by git, but will be imported
NOTES
  [none]
RESOURCES
  [none]
CONVENTIONS
  -- all directory strings will not include a trailing slash
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

# internal imports
import epglib.constants as c_epg


# data_pruning.py
patients_dp = {
               # 1: 90, 2: 90
               # 1: 90, 2: 90, 67: 90
               # 69: 90
               # 3: 90, 4: 90, 5: 90, 6: 90, 7: 90, 8: 90, 9: 90
               # 20: 90, 21: 90, 22: 90, 23: 90, 24: 90, 66: 90, 67: 90, 68: 90, 69: 90, 80: 90, 81: 90
               # 25: 90, 26: 90, 27: 90, 28: 90, 29: 90,
               # 30: 90, 31: 90, 32: 90, 33: 90, 34: 90, 35: 90, 36: 90, 37: 90, 38: 90, 39: 90,
               # 40: 90, 41: 90, 42: 90, 43: 90, 44: 90, 45: 90, 46: 90, 47: 90, 48: 90, 49: 90
               # 23: 93, 31: 51  # 31 = outlier?
               # 50: 90, 51: 90, 52: 90, 53: 90, 54: 90, 55: 90, 56: 90, 57: 90, 58: 90, 59: 90,
               # 60: 90, 61: 90, 62: 90, 63: 90, 64: 90, 65: 90,
               # 70: 90, 71: 90, 72: 90, 73: 90, 74: 90, 75: 90, 76: 90, 77: 90, 78: 90, 79: 90
               # 51: 19, 61: 91, 77: 91  # 51 = outlier?
               10: 90, 11: 90, 12: 90, 13: 90, 14: 90, 15: 90, 16: 90, 17: 90, 18: 90, 19: 90
              }  # keys = patient, values = latest-stage trial with all three conditions (default to <= 90)


# feature_gen.py

selected_condition = 1  # experimental modes, see data/my_meta.txt

# ComprehensiveFCParameters:
# patients_fg = [1, 2]  # 0:16:35, 55230 features, 1943 features with all zeros
# patients_fg = [1, 2, 67, 69]  # 0:31:08, 55230 features, 1922 features with all zeros
# patients_fg = range(59, 74+1)  # half 0, half 1; 2:07:17, 55230 features, 1870 features with all zeros
# runtime regression is very linear: T = 7.4*N + 1.66 => T(81) = 601min = 10 hours

# EfficientFCParameters:
# patients_fg = [1, 2]  # 0:01:14, 54810 features, 2202 features with all zeros
# patients_fg = [1, 2, 67, 69]  # 0:02:23, 54810 features, 2090 features with all zeros
# patients_fg = range(59, 74+1)  # half 0, half 1; 0:09:02, 54810 features, 2033 features with all zeros
# runtime regression is very linear: T = 0.556*N + 0.139 => T(81) = 45min
patients_fg = range(1, 81+1)  # full dataset; 0:44:38, 54810 features, 2021 features with all zeros

# fname_fgen = 'testing.csv'
# fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat1-2.csv'
# fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat1-2-67-69.csv'
# fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat59to74.csv'
fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat1to81.csv'


# feature_gen_post.py

eps_flat = 1e-6  # to flatten values to zero within |x| < eps_flat
test_size_fgp = 0.2  # ratio of test data to full dataset (see test_size)

# fname_fgen_post_in = fname_fgen
# fname_fgen_post_in = 'feature_gen_cond1_pat1-2.csv'  # no 'post' in name
# fname_fgen_post_out = 'feature_gen_post_cond1_pat1-2.csv'  # includes 'post' in name
# fname_fgen_post_in = 'feature_gen_cond1_pat1-2-67-69.csv'  # no 'post' in name
# fname_fgen_post_out = 'feature_gen_post_cond1_pat1-2-67-69.csv'  # includes 'post' in name
# fname_fgen_post_in = 'feature_gen_cond1_pat59to74.csv'  # no 'post' in name
# fname_fgen_post_out = 'feature_gen_post_cond1_pat59to74.csv'  # includes 'post' in name
fname_fgen_post_in = 'feature_gen_cond1_pat1to81.csv'  # no 'post' in name
fname_fgen_post_out = 'feature_gen_post_cond1_pat1to81.csv'  # includes 'post' in name


# run_PCA.py

# pca_data_handle = 'cond1_pat59to74'
pca_data_handle = 'cond1_pat1to81'
fname_pca_in = 'feature_gen_post_' + pca_data_handle + '.csv'
test_size = test_size_fgp  # test_size should be equal to the test_size_fgp corresponding to fname_pca_in
pca_show_fig = 'on'  # 'on' = run plt.show()
