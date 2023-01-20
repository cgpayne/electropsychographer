#!/usr/bin/env python
'''
classes.py = holds classes specific to the electropsychographer
  import epglib.classes as cls
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  these classes are distributed throughout the main files (eg, feature_gen.py)
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
import pandas as pd


class PatientDF():
    '''
    CLASS: PatientDF = a pandas data frame with properties for a patient sample
    '''
    # CONSTRUCTOR
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def tidy_up(self, header: list) -> None:
        '''
        METHOD: tidy_up = add a header and reassign data types in the df
        '''
        self.df.columns = header
        self.df['subject'] = self.df['subject'].astype('int')
        self.df['trial'] = self.df['trial'].astype('int')
        self.df['condition'] = self.df['condition'].astype('int')
        self.df['sample'] = self.df['sample'].astype('int')
    
    def set_condition(self, selected_condition: int) -> None:
        '''
        METHOD: set_condition = pull out a specific condition (experimental mode) from the df
            IN: selected_condition = the condition number to pull out, set in user_config.py
        '''
        self.df = self.df[self.df['condition'] == selected_condition]
    
    # OBSOLETE: now using only one trial via data_pruning.py
    # def line_up_time_series(self):
    #     '''
    #     METHOD: line_up_time_series = concatenate the time series in series wrt trial
    #     '''
    #     # concatenate all the time series together
    #     self.df[['sample', 'trial']] = \
    #         self.df[['sample', 'trial']].apply(lambda row: tss(row), axis=1)
    #
    #     # sort by sample
    #     self.df = self.df.sort_values(by=['sample'])
    #     self.df.reset_index(inplace=True, drop=True)
