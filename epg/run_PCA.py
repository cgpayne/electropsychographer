#!/usr/bin/env python
#  run_PCA.py = run the Principal Component Analysis (PCA)
#  python3 run_PCA.py
#  Copyright (c) 2022 Charlie Payne
#  Licence: GNU GPLv3
# DESCRIPTION
#  [insert: here]
# NOTES
#  [none]
# RESOURCES
#  -- https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
#  -- https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
# CONVENTIONS
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

# external imports
# import numpy as np
import pandas as pd

import epglib.params as p_epg
from config import user_config as cfg
# import epglib.utils as ut
# import epglib.classes as cls


if __name__ == '__main__':
    # create response variable
    demo = pd.read_csv(cfg.meta_dir + '/' + p_epg.fdemographic)
    print(demo)
    
    all_subjects = {demo.loc[ii, 'subject']: demo.loc[ii, ' group'] for ii in range(len(demo))}
    print(all_subjects)
    
    # df.index.name = 'subject' (for feature_gen.py)
    X = pd.DataFrame({'subject': [27,71,2,14,15,69,18,80], 'xyz': [88,89,90,91,92,9,93,94]})  # test
    y = pd.DataFrame({'subject': list(X['subject']),
                      'class': [all_subjects[X.loc[ii, 'subject']] for ii in range(len(X))]})
    print(y)
    
    
    # F- I-- N---
    print('\n\nF- I-- N---')
