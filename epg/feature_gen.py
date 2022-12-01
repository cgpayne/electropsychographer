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
#  -- make the patient data frames into a class?

# external imports
import csv
import pandas as pd
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute

# internal imports
from config import user_config as cfg


# FUNCTION: print_first_three = print the first three patient dataframes
def print_first_three(df_patients):
    for pp in list(df_patients.keys())[:3]:
        print(f"~~~~ patient {pp}, df =")
        print(df_patients[pp], '\n')


# FUNCTION: tss = put the time series in series wrt trial
#   IN/OUT: row = a row of the patient data frame
def tss(row):
    row['sample'] += (row['trial']-1)*cfg.sample_count
    return row


if __name__ == '__main__':
    # load in the data
    print("- loading in the data")
    df_patients = {}  # a dictionary of data frames per patient
    for pp in cfg.patients:
        print(f"  -- loading in for patient {pp}...")
        df_patients[pp] = pd.read_csv(cfg.archive + '/' + str(pp) + '.csv/' + str(pp) + '.csv', header=None)
        print("     ...done.")
    
    # grab and distribute the headers, separate out the selected_condition
    with open(cfg.archive + '/columnLabels.csv', 'r') as fin:
        heading = list(csv.reader(fin))[0]
    
    print("\n- setting the header and etc")
    print(f"  -- heading = {heading}\n")
    
    for pp in df_patients:
        df_patients[pp].columns = heading
        df_patients[pp]['subject'] = df_patients[pp]['subject'].astype('int')
        df_patients[pp]['trial'] = df_patients[pp]['trial'].astype('int')
        df_patients[pp]['condition'] = df_patients[pp]['condition'].astype('int')
        df_patients[pp]['sample'] = df_patients[pp]['sample'].astype('int')
        
        df_patients[pp] = df_patients[pp][df_patients[pp]['condition'] == cfg.selected_condition]
    
    print_first_three(df_patients)
    
    # put the waves in series
    print("- line up the trials into a series")
    for pp in df_patients:
        print(f"  -- lining up for patient {pp}...")
        df_patients[pp][['sample', 'trial']] = \
            df_patients[pp][['sample', 'trial']].apply(lambda row: tss(row), axis=1)
        
        df_patients[pp] = df_patients[pp].sort_values(by=['sample'])
        df_patients[pp].reset_index(inplace=True, drop=True)
        print("     ...done.")
    
    print_first_three(df_patients)


# F- I-- N---
