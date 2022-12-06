#!/usr/bin/env python
#  feature_gen.py = generate features from the time series using ts-fresh
#  python3 feature_gen.py
#  By: Charlie Payne
#  Licence: n/a
# DESCRIPTION
#  [TBA]
# NOTES
#  [none]
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  -- add a time stamp

# external imports
import time
import csv
import pandas as pd
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute

# internal imports
from config import user_config as cfg
import utils.utils as ut


# CLASS: PatientDF = a pandas data frame with properties for a patient sample
class PatientDF():
    # CONSTRUCTOR
    def __init__(self, df):
        self.df = df
    
    # METHOD: tidy_up = add a header and reassign data types in the df
    def tidy_up(self, header):
        self.df.columns = header
        self.df['subject'] = self.df['subject'].astype('int')
        self.df['trial'] = self.df['trial'].astype('int')
        self.df['condition'] = self.df['condition'].astype('int')
        self.df['sample'] = self.df['sample'].astype('int')
    
    # METHOD: set_condition = pull out a specific condition from the df
    #     IN: selected_condition = the condition number to pull out, set in user_config.py
    def set_condition(self, selected_condition):
        self.df = self.df[self.df['condition'] == selected_condition]
    
    # METHOD: line_up_time_series = concatenate the time series in series wrt trial
    def line_up_time_series(self):
        # concatenate all the time series together
        self.df[['sample', 'trial']] = \
            self.df[['sample', 'trial']].apply(lambda row: tss(row), axis=1)
        
        # sort by sample
        self.df = self.df.sort_values(by=['sample'])
        self.df.reset_index(inplace=True, drop=True)


# FUNCTION: print_first_three = print the first three patient data frames
def print_first_three(pat_dat):
    print("\n- printing first three data frames")
    for pp in list(pat_dat.keys())[:3]:
        print(f"~~~~ patient {pp}, df =")
        print(pat_dat[pp].df, '\n')


# FUNCTION: tss = put the time series in series wrt trial
#   IN/OUT: row = a row of the patient data frame
def tss(row):
    row['sample'] += (row['trial']-1)*cfg.sample_count
    return row


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    # load in the data
    print("- loading in the data")
    pat_dat = {}  # a dictionary of PatientDF's per patient
    for pp in cfg.patients:
        print(f"  -- loading in for patient {pp}...")
        pat_dat[pp] = PatientDF(pd.read_csv(cfg.archive + '/' + str(pp) + '.csv/' + str(pp) + '.csv', header=None))
        print("     ...done.")
        t_now = ut.time_stamp(t_now, t_zero, str(pp))  # TIME STAMP
    
    # grab the header, tidy up, and separate out the chosen condition
    with open(cfg.archive + '/columnLabels.csv', 'r') as fin:
        header = list(csv.reader(fin))[0]
    print(f"  -- header = {header}\n")
    
    print("- restructuring the data frames (tidy up + set condition)")
    for pp in pat_dat:
        pat_dat[pp].tidy_up(header)
        pat_dat[pp].set_condition(cfg.selected_condition)
    t_now = ut.time_stamp(t_now, t_zero, 'tidy up + set condition')  # TIME STAMP
    
    print_first_three(pat_dat)
    
    # put the waves in series
    print("- lining up the trials into a series")
    for pp in pat_dat:
        print(f"  -- lining up for patient {pp}...")
        pat_dat[pp].line_up_time_series()
        print("     ...done.")
        t_now = ut.time_stamp(t_now, t_zero, str(pp))  # TIME STAMP
    
    print_first_three(pat_dat)
    
    # concatenate the data and remove unnecessary columns
    print("- concatenating all the patient data")
    df_all_pats = pd.DataFrame({})
    for pp in pat_dat:
        df_all_pats = pd.concat([df_all_pats, pat_dat[pp].df])
    df_all_pats = df_all_pats.drop(columns=['condition', 'trial'])
    df_all_pats.reset_index(inplace=True, drop=True)
    
    print(df_all_pats)

# F- I-- N---
