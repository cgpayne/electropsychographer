#!/usr/bin/env python
#  user_config.py = holds parameters for the electropsychographer
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
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import os


# feature_gen.py
patients = [1, 2]
archive = os.getcwd() + '/../data/archive'
selected_condition = 1
sample_count = 3072
