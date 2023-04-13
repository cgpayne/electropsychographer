#!/usr/bin/env python
'''
run_PCA.py = run the Principal Component Analysis (PCA)
  python3 run_PCA.py <pca_mode> <pca_data_handle> <kfold_num> <pca_show_fig>
  Copyright (c) 2022 Charlie Payne
  Licence: GNU GPLv3
DESCRIPTION
  this takes in the output from feature_gen.py and splits it into training and test data
    then runs a PCA on it
    thus reducing the 55k generated features from before to on the order of 10
  data is taken in from c_epg.fgen_dir and saved to a sub-directory of c_epg.inter_dir
  run this after split_the_data.py and before notebooks/models/*.ipynb
NOTES
  [none]
RESOURCES
  -- https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
  -- https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
  -- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
CONVENTIONS
  [none]
KNOWN BUGS
  [none]
DESIRED FEATURES
  [none]
'''

# external imports
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# internal imports
import epglib.constants as c_epg
# from config import user_config as cfg
import epglib.utils as ut
from epglib.utils import eprint

pca_mode = sys.argv[1]  # 'pca' = for regular PCA, 'kpca' = for KernelPCA
pca_data_handle = sys.argv[2]  # eg, 'cond1_pat1to81_outrmv'
folding_num = sys.argv[3]  # the "folding" to run (eg, 0 -> folding with fold=0 as test and training as otherwise), use a any value for standard splitting
pca_show_fig = sys.argv[4]  # 'on' = run plt.show()


class DataEPG():
    '''
    CLASS: DataEPG = holds the design matrix and response and corresponding PCA methods
           methods are listed in order of operation
    '''
    # CONSTRUCTOR
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
        # do some book keeping
        y_train_count0 = len([sub for sub in y_train['class'] if sub == 0])
        y_train_count1 = len(y_train['class']) - y_train_count0
        y_test_count0 = len([sub for sub in y_test['class'] if sub == 0])
        y_test_count1 = len(y_test['class']) - y_test_count0
        print("- some book keepting:")
        print(f"  -- y_train has {y_train_count1} / {y_train_count0} = {y_train_count1/y_train_count0:.2f}x 1's to 0's")
        print(f"     y_test has {y_test_count1} / {y_test_count0} = {y_test_count1/y_test_count0:.2f}x 1's to 0's")
        print(f"     compared to 49 / 32 = {49/32:.2f}x 1's (SZ) to 0's (HC) in the full dataset")
        
        # set the members
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def print_Xy(self) -> None:
        '''
        METHOD: print_Xy = print X_train, X_test, y_train, then y_test
        '''
        print("\n  -- X_train =")
        print(self.X_train)
        print("\n  -- X_test =")
        print(self.X_test)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("\n  -- y_train =")
        print(self.y_train)
        print(f"#(0's) = {len([ii for ii in self.y_train['class'] if ii == 0])}, #(1's) = {len([ii for ii in self.y_train['class'] if ii == 1])}")
        print("\n  -- y_test =")
        print(self.y_test)
        print(f"#(0's) = {len([ii for ii in self.y_test['class'] if ii == 0])}, #(1's) = {len([ii for ii in self.y_test['class'] if ii == 1])}")
    
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
    
    def exec_PCA(self, pca_mode: str) -> None:
        '''
        METHOD: exec_PCA = execute the (kernel) PCA on X_train and X_test based on the X_train data
            IN: pca_mode = 'pca' for regular PCA, 'kpca' for KernelPCA
           OUT: << X_train and X_test altered, pass object by reference >>
        '''
        if pca_mode == 'pca':
            print("\n- perform the PCA")
            self.pca = PCA()
        elif pca_mode == 'kpca':
            print("\n- perform a kernel PCA")
            self.pca = KernelPCA(kernel='linear')
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
        # plt.xticks([ii for ii in range(1, self.num_components+1)])
        plt.xticks([1] + list(np.arange(5, 65, 5)))
        plt.savefig(fig_dir_now + '/explained_variance.pdf')
        if pca_show_fig == 'on':
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
        # plt.xticks([ii for ii in range(1, self.num_components+1)])
        plt.xticks([1] + list(np.arange(5, 65, 5)))
        plt.savefig(fig_dir_now + '/cummulative_explained_variance.pdf')
        if pca_show_fig == 'on':
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
            if self.y_train.loc[ii, 'class'] == 0:
                self.X_HC = np.append(self.X_HC, [self.X_train[ii, :]], axis=0)
            elif self.y_train.loc[ii, 'class'] == 1:
                self.X_SZ = np.append(self.X_SZ, [self.X_train[ii, :]], axis=0)
        
        # add in the test data
        # for ii in [2]:
        for ii in range(len(self.y_test)):
            if self.y_test.loc[ii, 'class'] == 0:
                self.X_HC = np.append(self.X_HC, [self.X_test[ii, :]], axis=0)
            elif self.y_test.loc[ii, 'class'] == 1:
                self.X_SZ = np.append(self.X_SZ, [self.X_test[ii, :]], axis=0)
    
    def plot_PC(self, pca_mode: str, fig_dir_now: str) -> None:
        '''
        METHOD: plot_PC = plot PC1 vs PC2 and PC2 vs PC3 for the HC and SZ data
            IN: pca_mode = 'pca' for regular PCA, 'kpca' for KernelPCA
                fig_dir_now = the directory that holds the generated figures
           OUT: << figures saved to fig_dir_now >>
        '''
        print("    --- ...PC1 vs PC2...")
        plt.figure()
        plt.scatter(self.X_HC[:, 0], self.X_HC[:, 1], label='HC')
        plt.scatter(self.X_SZ[:, 0], self.X_SZ[:, 1], label='SZ')
        plt.legend()
        plt.title('PC1 vs PC2')
        if pca_mode == 'pca':
            plt.xlabel(f"PC1 = {100*self.explained_variance[0]:.2f}%")
            plt.ylabel(f"PC2 = {100*self.explained_variance[1]:.2f}%")
        elif pca_mode == 'kpca':
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        plt.savefig(fig_dir_now + '/PC1_vs_PC2.pdf')
        if pca_show_fig == 'on':
            plt.show()
        
        print("    --- ...PC2 vs PC3...")
        plt.figure()
        plt.scatter(self.X_HC[:, 1], self.X_HC[:, 2], label='HC')
        plt.scatter(self.X_SZ[:, 1], self.X_SZ[:, 2], label='SZ')
        plt.title('PC2 vs PC3')
        if pca_mode == 'pca':
            plt.xlabel(f"PC2 = {100*self.explained_variance[1]:.2f}%")
            plt.ylabel(f"PC3 = {100*self.explained_variance[2]:.2f}%")
        elif pca_mode == 'kpca':
            plt.xlabel("PC2")
            plt.ylabel("PC3")
        plt.legend()
        plt.savefig(fig_dir_now + '/PC2_vs_PC3.pdf')
        if pca_show_fig == 'on':
            plt.show()
    
    def save(self, pca_mode: str, folding_num: int, cv_mode: str) -> None:
        '''
        METHOD: save = save the X's and y's to csv
           OUT: << csv's saved to a sub-directory of c_epg.inter_dir >>
          NOTE: must run set_HCSZ() before performing this method
        '''
        print("- saving X's and y's to csv")
        out_dir = c_epg.inter_dir + '/' + pca_mode + '/' + pca_data_handle
        ut.make_dir(out_dir)
        if cv_mode == 'kfold':
            the_folding = '-' + folding_num
        else:
            the_folding = ''
        self.y_train.to_csv(out_dir + '/y_train' + the_folding + '_' + pca_data_handle + '.csv')
        self.y_test.to_csv(out_dir + '/y_test' + the_folding + '_' + pca_data_handle + '.csv')
        np.savetxt(out_dir + '/X_train' + the_folding + '_' + pca_data_handle + '.csv', self.X_train, delimiter=',')
        np.savetxt(out_dir + '/X_test' + the_folding + '_' + pca_data_handle + '.csv', self.X_test, delimiter=',')
        np.savetxt(out_dir + '/X_HC' + the_folding + '_' + pca_data_handle + '.csv', self.X_HC, delimiter=',')
        np.savetxt(out_dir + '/X_SZ' + the_folding + '_' + pca_data_handle + '.csv', self.X_SZ, delimiter=',')


if __name__ == '__main__':
    # parse the input
    if pca_mode not in ['pca', 'kpca']:
        eprint("ERROR 756: invalid option for pca_mode!")
        eprint(f"pca_mode = {pca_mode}")
        eprint("valid options are: 'pca', 'kpca'")
        eprint("exiting...")
        sys.exit(1)
    if 'kfold' in pca_data_handle.split('_')[-1]:
        cv_mode = 'kfold'
    elif 'standard' in pca_data_handle.split('_')[-1]:
        cv_mode = 'standard'
    else:
        eprint("ERROR 180: invalid pca_data_handle!")
        eprint(f"pca_data_handle = {pca_data_handle}")
        eprint("it must be of the form *_standard or *_kfold-*")
        eprint("exiting...")
        sys.exit(1)
    
    ## set up for the PCA
    t_zero = time.time()  # start the clock
    t_now = t_zero
    
    print(f"- processing: pca_data_handle = {pca_data_handle}")
    print(f"              folding_num = {folding_num}\n")
    
    # load in the data
    data_sub_dir = c_epg.split_dir + '/' + pca_data_handle
    if cv_mode == 'kfold':
        fend = folding_num + '_' + pca_data_handle.split('_kfold')[0] + '.csv'
        fX_train = data_sub_dir + '/X_train-' + fend
        fX_test = data_sub_dir + '/X_test-' + fend
        fy_train = data_sub_dir + '/y_train-' + fend
        fy_test = data_sub_dir + '/y_test-' + fend
    else:
        fX_train = data_sub_dir + '/X_train_' + pca_data_handle + '.csv'
        fX_test = data_sub_dir + '/X_test_' + pca_data_handle + '.csv'
        fy_train = data_sub_dir + '/y_train_' + pca_data_handle + '.csv'
        fy_test = data_sub_dir + '/y_test_' + pca_data_handle + '.csv'
    
    print("- loading in the data")
    X_train = np.loadtxt(fX_train, delimiter=',')
    X_test = np.loadtxt(fX_test, delimiter=',')
    y_train = pd.read_csv(fy_train)
    y_test = pd.read_csv(fy_test)
    
    ## build the model
    epg = DataEPG(X_train, X_test, y_train, y_test)
    epg.print_Xy()
    t_now = ut.time_stamp(t_now, t_zero, 'train/test split')  # TIME STAMP
    
    # scale the data for optimized performance
    epg.scale_X()
    epg.print_X()
    t_now = ut.time_stamp(t_now, t_zero, 'scaled data')  # TIME STAMP
    
    ## perform the PCA and plot
    epg.exec_PCA(pca_mode)
    epg.print_X()
    t_now = ut.time_stamp(t_now, t_zero, 'PCA')  # TIME STAMP
    
    fig_dir_now = c_epg.fig_dir + '/' + pca_data_handle + '/' + str(folding_num)
    if pca_mode == 'pca':
        # plot the explained variance
        epg.set_ev()
        print("  -- plotting...")
        ut.make_dir(fig_dir_now)
        epg.plot_ev(fig_dir_now)
        
        # plot the cummulative explained variance
        epg.plot_cev(fig_dir_now)
    elif pca_mode == 'kernel':
        print("  -- plotting...")
    
    # split out training data into control (HC) and schizophrenia (SZ)
    epg.set_HCSZ()
    
    # plot PC1 vs PC2 then PC2 vs PC3
    epg.plot_PC(pca_mode, fig_dir_now)
    
    print("     ...done.")
    t_now = ut.time_stamp(t_now, t_zero, 'plotting')  # TIME STAMP
    
    # save to file
    epg.save(pca_mode, folding_num, cv_mode)
    t_now = ut.time_stamp(t_now, t_zero, 'save')  # TIME STAMP
    
    
    print('\n\nF- I-- N---')
