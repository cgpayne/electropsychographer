#!/usr/bin/env python
#  user_config.py = holds user configuration parameters for the electropsychographer
#  from config import user_config as cfg
#  By: Charlie Payne
#  Licence: n/a
# DESCRIPTION
#  user_config.py is the main configuration file
#  to set this up: copy example_config.py -> user_config.py
#  example_config.py will be tracked by git, but will *not* be imported
#  user_config.py will *not* be tracked by git, but will be imported
# NOTES
#  [none]
# RESOURCES
#  [none]
# CONVENTIONS
#  -- all directory strings will not include a trailing slash
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import os


# [shared]
data_dir = 'data'
archive_dir = os.getcwd() + '/../' + data_dir + '/sub_archive'
meta_dir = os.getcwd() + '/../' + data_dir + '/meta_archive'
pruned_dir = os.getcwd() + '/../' + data_dir + '/pruned'
inter_dir = os.getcwd() + '/../' + data_dir + '/intermediates'

# data_pruning.py
patients_dp = {
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
               51: 19, 61: 91, 77: 91  # 51 = outlier?
              }  # keys = patient, values = late-stage trial with all three conditions

# feature_gen.py
# patients_fg = [1, 2]  # 0:16:35, 55230 features, 1943 features with all zeros
# patients_fg = [1, 2, 67, 69]  # 0:31:08, 55230 features, 1922 features with all zeros
patients_fg = range(59, 74+1)  # half 0, half 1; 2:07:17, 55230 features, 1870 features with all zeros
# patients_fg = range(1, 81+1)
selected_condition = 1
# fname_fgen = 'feature_gen_pat1-2_cond' + str(selected_condition) + '.csv'
# fname_fgen = 'feature_gen_pat1-2-67-69_cond' + str(selected_condition) + '.csv'
fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat59to74.csv'
# fname_fgen = 'feature_gen_cond' + str(selected_condition) + '_pat1to81.csv'
