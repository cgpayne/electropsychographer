#!/usr/bin/env python
#  data_pruning.py = do an initial pruning, pull out the right trial for selected data
#  python3 data_pruning.py
#  Copyright (c) 2022 Charlie Payne
#  Licence: GNU GPLv3
# DESCRIPTION
#  the purpose of this script is to individually pull out a single trial (eg, 100)
#    from each of patient data frames
#  this is to save memory, and will be done in batches
#  it is very basic in that reformatting the data frames will be handled in feature_gen.py
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
import time
import pandas as pd

# internal imports
from config import user_config as cfg
import epglib.utils as ut


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- running for patients: {list(cfg.patients_dp.keys())}\n")
    
    for pp in cfg.patients_dp:
        # load in the data
        print(f"- operating on patient {pp}")
        print("  -- loading in the data...")
        df_pat = pd.read_csv(cfg.archive_dir + '/' + str(pp) + '.csv/' + str(pp) + '.csv', header=None)
        print("     ...done.")
        t_now = ut.time_stamp(t_now, t_zero, 'data load in')  # TIME STAMP
        # exit()
        
        # check length of data frame for all other trials, looking for 9216 = 3*sample_count
        other_trials = {val: len(df_pat[df_pat[1] == val]) for val in range(100+1)}
        
        # select a single trial
        print(f"  -- selecting trial {cfg.patients_dp[pp]}")
        df_pat = df_pat[df_pat[1] == cfg.patients_dp[pp]]
        df_pat.reset_index(inplace=True, drop=True)
        
        # throw warning with all three conditions not present for this trial
        check_warning = False
        for ii in range(1, 3+1):
            print('')
            if df_pat[df_pat[2] == ii].empty:
                print(f"!!!!---->>>> WARNING: patient {pp} is missing condition = {ii} <<<<----!!!!")
                check_warning = True
        if check_warning:
            print(f"!!!!---->>>> try other trials: {other_trials}")
        print("\n    --- df_pat =")
        print(df_pat)
        
        # output to csv
        print("  -- saving to file")
        df_pat.to_csv(cfg.pruned_dir + '/' + str(pp) + '.csv', index=False, header=False)
        t_now = ut.time_stamp(t_now, t_zero, 'selection, check, save')  # TIME STAMP
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
