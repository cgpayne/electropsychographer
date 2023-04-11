# Introduction

### Goal

The aim of this project is to predict whether or not a patient has schizophrenia based on their electroencephalogram (EEG) reading using a machine learning model.

### Motivation

Schizophrenia is a devastating mental health disorder which affects approximately [0.7% of the population](https://academic.oup.com/epirev/article/30/1/67/621138?login=false), meaning there are an estimated 56 million people with schizophrenia worldwide. This disorder can lead to [significant disability](https://pubmed.ncbi.nlm.nih.gov/11153962/), and an [estimated 6%](https://www.neomed.edu/medicine/wp-content/uploads/sites/2/Suicide-Risk-in-First-Episode-Psychosis-A-Selective-Review-of-Current-Literature.pdf) of people with schizophrenia will commit suicide in their lifetime - a 6x higher rate than the [US national average](https://afsp.org/suicide-statistics/) - with [22% attempting suicide](https://www.sciencedirect.com/science/article/abs/pii/S0920996409004964?via%3Dihub) within 7.4 years of the first episode of psychosis.

Treatment options are limited to antipsychotic medications, which carry a high burden of [side effects](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3384511/), and varying effectivity across the patient population. Nevertheless, access to treatment in a timely manner is a crucial step in the early intervention of psychotic-type illnesses, since effective treatment regimes can help a patient lead a fulfilling life.

However, a diagnosis of schizophrenia can only be made by a psychiatrist, and patients typically go [74 weeks](https://ajp.psychiatryonline.org/doi/full/10.1176/appi.ajp.2015.15050632) (1.4 years) from onset of initial symptoms to treatment with a prescribed antipsychotic. This is in stark contrast to a disease like cancer, which typically takes several weeks to diagnose. The difference is that schizophrenia currently has no physiological test - that is, there is no "litmus test" for schizophrenia, or for any known mental disorder. Establishing such tests is a major objective in the modern treatment of mental illness.

Access to a cost effective, non-invasive, rapid test for the diagnosis of patients with schizophrenia would be a major breakthrough in the field. An EEG, which records electrical activity of the brain through nodes on the head, could provide a viable candidate for a testing procedure. The average cost of an EEG reading in the United States is [\$200](https://www.researchgate.net/profile/Richard-Lipton/publication/230131715_Cost_of_Health_Care_Among_Patients_With_Chronic_and_Episodic_Migraine_in_Canada_and_the_USA_Results_From_the_International_Burden_of_Migraine_Study_IBMS/links/5e0a72224585159aa4a6eee2/Cost-of-Health-Care-Among-Patients-With-Chronic-and-Episodic-Migraine-in-Canada-and-the-USA-Results-From-the-International-Burden-of-Migraine-Study-IBMS.pdf), the procedure involves the non-invasive placement of painless electrodes on the head, and software like a machine learning algorithm can provide a rapid evaluation of an EEG.

### Solution

Here, I propose that an EEG reading provides enough information to predict whether or not a patient has schizophrenia within a high degree of accuracy. This would act as a useful augmentation strategy for a psychiatrist's diagnostic procedure. To accomplish this, I will build a machine learning model on EEG data from a population of patients with and without schizophrenia.

# Data <a id="data"></a>

[insert: study, outline with links to Kaggle; condition, trial, time-points]

# Code

All the software is written in Python scripts (located in `epg/`) and Jupyter notebooks (located in `notebooks/models/`). Some notable packages used are: `ts-fresh`, `scikit-learn`, and `pytorch`. The Python version I use is `3.9.15` and there is a `requirements.txt` in the top directory.

There are two main phases of the software: the *script phase* and the *notebook phase*. In the *script phase* I handle the data processing in a production-ready manner, which involves reformatting the data, generating features from the time series (using `ts-fresh`), removing outliers, creating folds for cross validation, and finally running a PCA for dimensionality reduction. In the *notebook phase* I build the model in an exploratory manner, which uses the random forest algorithm that's optimized via a randomized search in a Jupyter notebook.

### Script Phase

This phase uses scripts under the `epg/` directory. After setting up the data (see [Data](#data)), I will follow these steps: data pruning, feature generation, outlier detection, PCA, and finally splitting the data into folds.

##### User Config

To run the respect scripts, I use a user configuration file to set certain parameters about the run. To see an example of these parameters and how they relate to the individual scripts, refer to the `epg/config/example_config.py` file. In order to execute a script, one must copy this example into a file called `user_config.py` where the parameters can be toggled for the run.

`cp example_config.py user_config.py`

Note that `example_config.py` is tracked by git, whereas `user_config.py` is not, as seen in the `.gitignore` file.

##### Stages

I will now describe the individual scripts, their purpose, and how to run them. Each script can be found under the `epg/` directory, and they all make use of the `user_config.py`.

###### 1 - Data Pruning

To reduce computational load, I work on only one trial out of the 100 trials per patient per condition. I assume that that later trials (which each represent a time series) will have the bigger difference in signature, since that's when the brain will be best trained against the tone in the button test. That is, patients without schizophrenia will now have a suppression of the neural activity from the predicted sound, whereas those with schizophrenia will not, in theory. So, I prune out the latest trial possible, where all three conditions are present, per patient. This restriction typically happens at trial 90, but sometimes lower or higher, but (for unexplained reasons) never at the final trial (ie, 100). I found the respective trials manually.

To run this, first set the `patients_dp` parameter in the `user_config.py`, which lists the patient dataset to be pruned and the trial to pull out. Then execute:

`python data_pruning.py`

The data is taken from the `archive_dir`, processed, and saved to the `pruned_dir` variables, which are set respectively in `epg/epglib/constants.py` file.

###### 2 - Feature Generation

I now use `ts-fresh` to generate a large range of features from the time series via the `epg/feature_gen.py` script and then some post-processing using the `epg/feature_gen_post.py` script.

Set the config parameters, and then run:

`python feature_gen.py`

For the full dataset of all the patients, this will produce roughly 54,810 features from the time series in under an hour using the `EfficientFCParameters` setting from `ts-fresh`. With more computational resources, one may also use the `ComprehensiveFCParameters` to produce an additional 420 more features.

There are many features which have all zeros exactly and also approximately (see the `eps_flat` parameter in the `user_config.py`). I excise those in `epg/feature_gen_post.py`, and also excise any features where there would be all zeros in the training data (see the `test_size_fpg` parameter in the `user_config.py`). I also impute the data overall. Run:

`python feature_gen_post.py`

###### 3 - Outlier Removal

[insert: describe after updating the method]

###### 4 - Split the Data for K-fold CV

I now split the data into the desired number of folds for cross validation (or into a single standard test/training split if desired) using the `epg/split_the_data.py` script. I choose a stratified K-fold cross validation to retain the class imbalance present between the 32 health controls to 49 schizophrenia patients.

Set the `user_config.py` parameters (make sure the `Kfolds` parameter matches with the `test_size_fgp` parameters from the prior `feature_gen_post.py` run), and then run:

`python split_the_data.py kfold`

Note that you may use any other string in place of `kfold` to run a standard `test_train_split` as compared to the `StratifiedKFold` (both from `sklearn.model_selection`). The data is taken from the `fgen_dir`, processed, then saved to the `split_dir` (see the `epg/epglib/constants.py` file) in form `X_train-*_<data_handle>.csv`, where the `*` wildcard refers to the fold number and the `<data_hanlde>` is set in the `user_config.py`.

###### 5 - PCA

To reduce the $~10^4$ features generated from the time series to a more computationally efficient dimensionality, I use Principal Component Analysis (PCA). This will pay off when running models like random forest and alike, but I must make sure I capture enough of the data's variance in a reasonable amount of principal components. To see the typical results of this PCA, refer to [Model and Results](#model-and-results) - I can reduce the dimension to $~10$. To run, try:

`python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 0 on`

where the `pca` can be changed with `kpca` for a Kernel-PCA, the `cond1_pat1to81_outrmv_kfold-5` can be any generic data handle which matches your naming scheme, `0` refers to the fold number, and `on` runs with the plots displayed during runtime. To run many folds, one can use a bash script such as:

```
pca_show_fig=${1}

python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 0 $pca_show_fig
python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 1 $pca_show_fig
python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 2 $pca_show_fig
python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 3 $pca_show_fig
python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 4 $pca_show_fig
```

Note that I also ran for a Kernel-PCA, which gave similar results for this dataset.

### Notebook Phase

[insert: stuff]

### Running the Code - Summary

[insert: description]

# Model and Results <a id="model-and-results"></a>

[insert: stuff]

### Results

[insert: stuff]

# Conclusion and Future Work

[insert: stuff]