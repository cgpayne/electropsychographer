#!/usr/bin/env python
#  params.py = holds hard-coded parameters for the electropsychographer
#  import epglib.params as p_epg
#  Copyright (c) 2022 Charlie Payne
#  Licence: GNU GPLv3
# DESCRIPTION
#  these parameters are hard coded, in contrast to config/user_config.py
# NOTES
#  [none]
# RESOURCES
#  [none]
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

# external imports
import os


data_dir = '../data'
archive_dir = os.getcwd() + '/' + data_dir + '/sub_archive'
meta_dir = os.getcwd() + '/' + data_dir + '/meta_archive'
pruned_dir = os.getcwd() + '/' + data_dir + '/pruned'
inter_dir = os.getcwd() + '/' + data_dir + '/intermediates'
fig_dir = '../figures'

sample_count = 3072  # number of samples, see data/archive/time.csv
fcol_labels = 'columnLabels.csv'
fdemographic = 'demographic.csv'
