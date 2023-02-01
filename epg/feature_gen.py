#!/usr/bin/env python
'''
feature_gen.py = generate features from the time series using ts-fresh
  python3 feature_gen.py
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  this code is the work horse of the electropsychographer
  it takes in the pruned data, concatenates it, then generates features of the
    time series using a package called ts-fresh
  there tends to be around 55,230 features generated, which are reduced via
    PCA in the script (run_PCA.py) following feature_gen_post.py
  data is taken in from c_epg.pruned_dir, processed, and saved to c_epg.inter_dir
NOTES
  [none]
RESOURCES
  -- https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
CONVENTIONS
  [none]
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

# external imports
import time
import csv
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut
import epglib.classes as cls


def print_first_three(pat_dat: cls.PatientDF) -> None:
    '''
    FUNCTION: print_first_three = print the first three patient data frames
    '''
    print("\n- printing first three data frames")
    for pp in list(pat_dat.keys())[:3]:
        print(f"~~~~ patient {pp}, df =")
        print(pat_dat[pp].df, '\n')


# OBSOLETE: now using only one trial via data_pruning.py
# def tss(row):
#     '''
#     FUNCTION: tss = put the time series in series wrt trial
#       IN/OUT: row = a row of the patient data frame
#     '''
#     row['sample'] += (row['trial']-1)*c_epg.sample_count
#     return row


if __name__ == '__main__':
    ## manipulate the data into the desired form
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- running for patients: {list(cfg.patients_fg)}\n")
    
    # load in the data
    print("- loading in the data")
    pat_dat = {}  # a dictionary of PatientDF's per patient
    for pp in cfg.patients_fg:
        print(f"  -- loading in for patient {pp}...")
        pat_dat[pp] = cls.PatientDF(pd.read_csv(c_epg.pruned_dir + '/' + str(pp) + '.csv', header=None))
        print("     ...done.")
        t_now = ut.time_stamp(t_now, t_zero, str(pp))  # TIME STAMP
    
    # grab the header, tidy up, and separate out the chosen condition
    with open(c_epg.meta_dir + '/' + c_epg.fcol_labels, 'r') as fin:
        header = list(csv.reader(fin))[0]
    print(f"  -- header = {header}\n")
    
    print("- restructuring the data frames (tidy up + set condition)")
    for pp in pat_dat:
        pat_dat[pp].tidy_up(header)
        pat_dat[pp].set_condition(cfg.selected_condition)
    t_now = ut.time_stamp(t_now, t_zero, 'tidy up + set condition')  # TIME STAMP
    
    print_first_three(pat_dat)
    
    # OBSOLETE: now using only one trial via data_pruning.py
    # # put the waves in series
    # print("- lining up the trials into a series")
    # for pp in pat_dat:
    #     print(f"  -- lining up for patient {pp}...")
    #     pat_dat[pp].line_up_time_series()
    #     print("     ...done.")
    #     t_now = ut.time_stamp(t_now, t_zero, str(pp))  # TIME STAMP
    #
    # print_first_three(pat_dat)
    
    # concatenate the data and remove unnecessary columns
    print("- concatenating all the patient data")
    df_all_pats = pd.DataFrame({})
    for pp in pat_dat:
        df_all_pats = pd.concat([df_all_pats, pat_dat[pp].df])
    df_all_pats = df_all_pats.drop(columns=['condition', 'trial'])
    df_all_pats.reset_index(inplace=True, drop=True)
    
    print("  -- df_all_pats =")
    print(df_all_pats, '\n')
    t_now = ut.time_stamp(t_now, t_zero, "concatenation")  # TIME STAMP
    
    ## feature generation using ts-fresh
    print("- generating the features using ts-fresh...")
    df_extracted = extract_features(df_all_pats, column_id="subject", column_sort="sample")
    print("  ...done.")
    print("  -- df_extracted =")
    print(df_extracted, '\n')
    
    # output to csv
    print("- saving to file")
    ut.make_dir(c_epg.inter_dir)
    df_extracted.index.name = 'subject'
    df_extracted.to_csv(c_epg.inter_dir + '/' + cfg.fname_fgen)
    t_now = ut.time_stamp(t_now, t_zero, 'feature generation, save')  # TIME STAMP


    # F- I-- N---
    print('\n\nF- I-- N---')
