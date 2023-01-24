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
# import time
# import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# internal imports
# import epglib.constants as c_epg
from config import user_config as cfg
# import epglib.utils as ut


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


class DataEPG():
    '''
    CLASS: DataEPG = holds the design matrix and response and corresponding PCA methods
           methods are listed in order of operation
    '''
    # CONSTRUCTOR
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        # split the data: training, testing
        print("- split the dataset into training data and test data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=math.floor(time.time()))
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def print_Xy(self) -> None:
        '''
        METHOD: print_Xy = print X_train, y_train, X_test, then y_test
        '''
        print("\n  -- X_train =")
        print(self.X_train)
        print("\n  -- y_train =")
        print(self.y_train)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("\n  -- X_test =")
        print(self.X_test)
        print("\n  -- y_test =")
        print(self.y_test)
    
    def print_X(self) -> None:
        '''
        METHOD: print_X = print X-train then X_test
        '''
        print("\n  -- X_train =")
        print(self.X_train)
        print("\n  -- X_test =")
        print(self.X_test)
    
    def scale_X(self) -> None:
        '''
        METHOD: scale_X = Z-score X_train and X_test based on the X_train parameters
           OUT: << X_train and X_test altered, pass object by reference >>
        '''
        print('- scale the data via Z-score')
        sc = StandardScaler()  # scale via Z-score
        self.X_train = sc.fit_transform(self.X_train)  # fit and transform the X_train data via Z-score
        self.X_test = sc.transform(self.X_test)        # transform the X_test data using the mean and standard deviation fit from X_train
        
    def exec_PCA(self) -> None:
        '''
        METHOD: exec_PCA = execute the PCA on X_train and X_test based on the X_train data
           OUT: << X_train and X_test altered, pass object by reference >>
        '''
        print("\n- perform the PCA")
        self.pca = PCA()
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)
    
    def set_ev(self) -> None:
        '''
        METHOD: set_ev = set the explained_variance, num_components, and cummulative_ev variables
        '''
        self.explained_variance = self.pca.explained_variance_ratio_
        self.num_components = len(self.explained_variance)
        print(f"  -- explained_variance = {self.explained_variance}")
        print(f"  -- number of components = {self.num_components}")
        
        sum_cev = 0
        self.cummulative_ev = []
        for ii in range(self.num_components):
            sum_cev += self.explained_variance[ii]
            self.cummulative_ev.append(sum_cev)
        print(f"  -- cummulative_ev = {self.cummulative_ev}")
    
    def plot_ev(self, fig_dir_now: str) -> None:
        '''
        METHOD: plot_ev = plot the explained variance
            IN: fig_dir_now = the directory that holds the generated figures
            OUT: << figures saved to fig_dir_now >>
        '''
        print("    --- ...explained variance...")
        plt.figure()
        plt.bar(range(1, self.num_components+1), self.explained_variance)
        plt.title('explained variance')
        plt.xlabel('component')
        plt.ylabel('percent explained')
        plt.xticks([ii for ii in range(1, self.num_components+1)])
        plt.savefig(fig_dir_now + '/explained_variance.pdf')
        if cfg.pca_show_fig == 'on':
            plt.show()
    
    def plot_cev(self, fig_dir_now: str) -> None:
        '''
        METHOD: plot_ev = plot the *cummulative* explained variance
            IN: fig_dir_now = the directory that holds the generated figures
            OUT: << figures saved to fig_dir_now >>
        '''
        print("    --- ...cummulative explained variance...")
        plt.figure()
        plt.bar(range(1, self.num_components+1), self.cummulative_ev)
        plt.title('cummulative explained variance')
        plt.xlabel('component')
        plt.ylabel('percent explained')
        plt.xticks([ii for ii in range(1, self.num_components+1)])
        plt.savefig(fig_dir_now + '/cummulative_explained_variance.pdf')
        if cfg.pca_show_fig == 'on':
            plt.show()
    
    def set_HCSZ(self) -> None:
        '''
        METHOD: set_HCSZ = separate the X_train into healthy controls (HC) and schizophrenia (SZ) patients
           OUT: << assign X_HC and X_SZ to self >>
        '''
        self.X_HC = np.empty((0, self.X_train.shape[1]))
        self.X_SZ = np.empty((0, self.X_train.shape[1]))
        
        # add in the training data
        for ii in range(len(self.y_train)):
            if self.y_train.iloc[ii] == 0:
                self.X_HC = np.append(self.X_HC, [self.X_train[ii, :]], axis=0)
            elif self.y_train.iloc[ii] == 1:
                self.X_SZ = np.append(self.X_SZ, [self.X_train[ii, :]], axis=0)
        
        # add in the test data
        for ii in range(len(self.y_test)):
        # for ii in [2]:
            if self.y_test.iloc[ii] == 0:
                self.X_HC = np.append(self.X_HC, [self.X_test[ii, :]], axis=0)
            elif self.y_test.iloc[ii] == 1:
                self.X_SZ = np.append(self.X_SZ, [self.X_test[ii, :]], axis=0)
    
    def plot_PC(self, fig_dir_now: str) -> None:
        '''
        METHOD: plot_PC = plot PC1 vs PC2 and PC2 vs PC3 for the HC and SZ data
            IN: fig_dir_now = the directory that holds the generated figures
           OUT: << figures saved to fig_dir_now >>
        '''
        print("    --- ...PC1 vs PC2...")
        plt.figure()
        plt.scatter(self.X_HC[:, 0], self.X_HC[:, 1], label='HC')
        plt.scatter(self.X_SZ[:, 0], self.X_SZ[:, 1], label='SZ')
        plt.legend()
        plt.title('PC1 vs PC2')
        plt.xlabel(f"PC1 = {100*self.explained_variance[0]:.2f}%")
        plt.ylabel(f"PC2 = {100*self.explained_variance[1]:.2f}%")
        plt.savefig(fig_dir_now + '/PC1_vs_PC2.pdf')
        if cfg.pca_show_fig == 'on':
            plt.show()
        
        print("    --- ...PC2 vs PC3...")
        plt.figure()
        plt.scatter(self.X_HC[:, 1], self.X_HC[:, 2], label='HC')
        plt.scatter(self.X_SZ[:, 1], self.X_SZ[:, 2], label='SZ')
        plt.title('PC2 vs PC3')
        plt.xlabel(f"PC2 = {100*self.explained_variance[1]:.2f}%")
        plt.ylabel(f"PC3 = {100*self.explained_variance[2]:.2f}%")
        plt.legend()
        plt.savefig(fig_dir_now + '/PC2_vs_PC3.pdf')
        if cfg.pca_show_fig == 'on':
            plt.show()
