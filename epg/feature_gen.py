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
#  [none]

# external imports
# import csv
import pandas as pd
# import matplotlib.pyplot as plt
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import impute

# internal imports
from config import user_config as cfg


if __name__ == '__main__':
    print("- loading in the data")
    df_patients = {}  # a dictionary of data frames per patient
    for ii in cfg.patients:
        print(f"  -- loading in for patient {ii}...")
        df_patients[ii] = pd.read_csv(cfg.archive + '/' + str(ii) + '.csv/' + str(ii) + '.csv', header=None)
        print("     ...done.")
    
    for key in df_patients:
        print(key)
        print(df_patients[key])
