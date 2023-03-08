# external imports
import time
import math
import pandas as pd
from scipy.stats import zscore

# internal imports
import epglib.constants as c_epg
from config import user_config as cfg
import epglib.utils as ut


class RemovalOutliers():
    def __init__(self, df, std_cutoff, hit_frac):
        self.X = X
        self.std_cutoff = std_cutoff
        self.hit_num = math.floor(hit_frac*self.X.shape[1])  # the total number of features outside std_cutoff to define an outlier
        
        print(f"  hit_num = {self.hit_num}")
    
    def find_outliers(self):
        # z-score the dataframe
        dfz = self.X.apply(zscore)
        self.X['hits'] = (dfz >= self.std_cutoff).sum(axis=1)
        
        self.outliers = list(self.X[self.X['hits'] >= self.hit_num].index)
        print(f"     we've found the outliers: {self.outliers}")

    def remove_outliers(self):
        # keep the samples which are not outliers, hence self.X['hits'] >= self.hit_num is False
        self.X = self.X[self.X['hits'] < self.hit_num].drop(columns=['hits'])
    
    def save(self):
        print("- saving X to csv")
        ut.make_dir(c_epg.fgen_dir)
        self.X.to_csv(c_epg.fgen_dir + '/' + cfg.fname_removal_out)


if __name__ == '__main__':
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- processing: {cfg.fname_removal_in}, with std_cutoff = {cfg.std_cutoff} and hit_frac = {cfg.hit_frac}\n")
    
    # load in the data
    print("- loading in the data")
    X = pd.read_csv(c_epg.fgen_dir + '/' + cfg.fname_removal_in, index_col='subject')
    t_now = ut.time_stamp(t_now, t_zero, 'load data')  # TIME STAMP
    
    # find and remove all outliers
    print("- finding and removing all outliers...")
    ro = RemovalOutliers(X, cfg.std_cutoff, cfg.hit_frac)
    print("  -- finding outliers...")
    ro.find_outliers()
    t_now = ut.time_stamp(t_now, t_zero, 'find outliers')  # TIME STAMP
    print("  -- removing outliers...")
    ro.remove_outliers()
    t_now = ut.time_stamp(t_now, t_zero, 'remove outliers')  # TIME STAMP
    print("  ...done.\n")
    
    # save to file
    ro.save()
    t_now = ut.time_stamp(t_now, t_zero, 'save')  # TIME STAMP
    
    
    print('\n\nF- I-- N---')
