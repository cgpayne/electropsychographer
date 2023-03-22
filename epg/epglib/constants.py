#!/usr/bin/env python
'''
constants.py = holds hard-coded constants for the electropsychographer
  import epglib.constants as c_epg
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  these constants are hard coded, in contrast to config/user_config.py
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
import os


data_dir = '../data'
archive_dir = os.getcwd() + '/' + data_dir + '/sub_archive'
meta_dir = os.getcwd() + '/' + data_dir + '/meta_archive'
pruned_dir = os.getcwd() + '/' + data_dir + '/pruned'
inter_dir = os.getcwd() + '/' + data_dir + '/intermediates'
fig_dir = '../figures'
fgen_dir = inter_dir + '/feature_gen'
split_dir = inter_dir + '/data_splittings'

sample_count = 3072  # number of samples, see data/archive/time.csv
fcol_labels = 'columnLabels.csv'
fdemographic = 'demographic.csv'
