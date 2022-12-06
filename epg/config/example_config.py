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
# CONVENTIONS
#  -- all directory strings will not include a trailing slash
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import os


# [shared]
archive_dir = os.getcwd() + '/../data/sub_archive'
pruned_dir = os.getcwd() + '/../data/pruned'

# data_pruning.py
patients_dp = {
               1: 90, 2: 90, 67: 90,
              }  # keys = patient, values = late-stage trial with all three conditions

# feature_gen.py
patients_fg = [1, 2, 67]
selected_condition = 1
