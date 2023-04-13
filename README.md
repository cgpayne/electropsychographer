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

We retrieved data from a [study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4059422/) of health controls (HC) versus patients with schizophrenia (SZ), from a dataset in two parts on Kaggle [here](https://www.kaggle.com/datasets/broach/button-tone-sz?resource=download) and [here](https://www.kaggle.com/datasets/broach/buttontonesz2). There were 32 HC's and 49 SZ's, for a total of 81 samples.

In this study, it is described that HC's have a mechanism in the brain which represses neural activity from expected or internal stimuli. For instance, if one pushes a button that emits a tone, and that tone is expected by the brain, the brain will suppress neural activity associated with the auditory stimulus of the tone. This allows for more efficient functioning in the sensory cortex. However, patients that have schizophrenia are known to have less suppression of this neural activity, which signifies a difference in processing external and internal stimuli.

It is posited that this dysfunction in neural suppression contributes to the experience of sensory hallucinations in patients with schizophrenia. Furthermore, the authors expect that one can characterize this neural activity on an EEG reading through 70 nodes on the patient's head which read the electrical activity of the brain.

I therefore asked the question: **is there enough of a signature in these EEG readings to distinguish between a healthy control and a patient with schizophrenia?** If so, can a machine learning model be trained on this difference, and then predict whether or not a new patient has schizophrenia based on their EEG? A model like this could be a major tool to psychiatrists as a physiological test (of which none currently exists) for schizophrenia.

In the study, a button was repeatedly pressed which emitted a tone and the EEG read the brain's electrical activity as it reacted to the stimulus of the noise. In a HC, as the button is pressed in succession, the brain should expect the tone and suppress the related neural activity, whereas a SZ will not. There were three conditions tested: the button press with a subsequent tone, the button press without a tone (silence), and simply a played tone without a button press involved. Each condition had an intended 100 trials (ie, 100 presses/tones), and each trial had 3072 time points measured on the EEG, which gives us a time series for each of the 70 nodes on the head. We chose to train on each condition separately to see which one had the best signature.

# Code

In this section we describe the structure of the code and how to run it in great detail. If you would like to see a summary of the model and the subsequent results, you can skip directly to [Model and Results](#model-and-results).

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

###### 1 - Data Pruning <a id="data-pruning"></a>

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

###### 4 - Split the Data for Stratified K-fold CV

I now split the data into the desired number of folds for cross validation (or into a single standard test/training split if desired) using the `epg/split_the_data.py` script. I choose a stratified K-fold cross validation to retain the class imbalance present between the 32 health controls to 49 schizophrenia patients.

Set the `user_config.py` parameters (make sure the `Kfolds` parameter matches with the `test_size_fgp` parameters from the prior `feature_gen_post.py` run), and then run:

`python split_the_data.py kfold`

Note that you may use any other string in place of `kfold` to run a standard `test_train_split` as compared to the `StratifiedKFold` (both from `sklearn.model_selection`). The data is taken from the `fgen_dir`, processed, then saved to the `split_dir` (see the `epg/epglib/constants.py` file) in form `X_train-*_<data_handle>.csv`, where the `*` wildcard refers to the fold number and the `<data_hanlde>` is set in the `user_config.py`.

###### 5 - PCA <a id="PCA"></a>

To reduce the ~$10^4$ features generated from the time series to a more computationally efficient dimensionality, I use Principal Component Analysis (PCA). This will pay off when running models like random forest and alike, but I must make sure I capture enough of the data's variance in a reasonable amount of principal components. To see the typical results of this PCA, refer to [Model and Results](#model-and-results) - I can reduce the dimension to ~$10$. To run, try:

`python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 0 on`

where the `pca` can be changed with `kpca` for a Kernel-PCA, the `cond1_pat1to81_outrmv_kfold-5` can be any generic data handle which matches your naming scheme, `0` refers to the fold number, and `on` runs with the plots displayed during runtime. To run many folds, one can use a bash script such as:

```
cond_num=${1}
pca_show_fig=${2}

python run_PCA.py pca cond${cond_num}_pat1to81_outrmv_kfold-5 0 $pca_show_fig
python run_PCA.py pca cond${cond_num}_pat1to81_outrmv_kfold-5 1 $pca_show_fig
python run_PCA.py pca cond${cond_num}_pat1to81_outrmv_kfold-5 2 $pca_show_fig
python run_PCA.py pca cond${cond_num}_pat1to81_outrmv_kfold-5 3 $pca_show_fig
python run_PCA.py pca cond${cond_num}_pat1to81_outrmv_kfold-5 4 $pca_show_fig
```

Note that I also ran for a Kernel-PCA, which gave similar results for this dataset.

### Notebook Phase

This is where the machine learning models are built and run. For a random forest using a stratified K-fold cross validation see the `random_forest_all.ipynb` notebook. In order to execute this notebook, all the data must be processed from [Stage 1](#data-pruning) through to [Stage 5](#PCA) in the *script phase* of the codebase. For the results from this notebook, see the [Model and Results](#model-and-results) section.

[insert: more]

### Running the Code - Summary

To run the code from start to finish, use submissions like:

```
python data_pruning.py
python feature_gen.py
python feature_gen_post.py
python outlier_removal.py
python split_the_data.py kfold
python run_PCA.py pca cond1_pat1to81_outrmv_kfold-5 0 on
```

making sure to set the `user_config.py` parameters before each submission. Then execute through the cells of the `notebooks/models/random_forest_all.py` Jupyter notebook.

# Model and Results <a id="model-and-results"></a>

In totality: the model used `ts-fresh` with `EfficientFCParamteres` to generate 54,810 features from the EEG time series, per condition; then we used a PCA via `sklearn.decomposition`to reduce the dimensionality to 60 principal components which capture 100% of the variance, per 5-fold for cross validation; and finally we used a random forest model `from sklearn.ensemble import RandomForestClassifier`, which we optimized using a random search, in `notebooks/models/random_forest_all.ipynb`.

In a typical PCA, we reduced to 60 principal components. For condition 2 fold 0, for example, the cumulative explained variance went as follows:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond2_fold0_pca/cummulative_explained_variance.png)

Somewhat ominously, there was no separation between healthy controls (HC) and patients with schizophrenia (SZ) in the PCA. However, we see that the first two principle components only capture 9.11% and 7.01% of the total variance:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond2_fold0_pca/PC1_vs_PC2.png)

We then chose to start with a random forest model on these 60 principal components since it is robust and more efficient than a neural net for non-linear structured data. In the future we will also train a linear regression and a neural net for comparison.

### Preliminary Results

To assess the random forest model, we used a stratified 5-fold cross validation, and averaged the following metrics for analysis: accuracy, precision, recall, and F1-score (the harmonic mean of the precision and recall). Since the data is unbalanced, we focus on the F1-score and use the `RandomForestClassifier(class_weight="balanced")` balanced classifier, and used a stratified K-fold CV to keep the ratio of HC to SZ consistent. To optimize the hyper-parameters of the random forest (that is, the size of the random forest and the depth of the decision trees), we used the `RandomizedSearchCV` from `sklearn.model_selection`.

As a control, we wanted to beat a model which predicted all 1's (ie, SZ's, which the data was unbalanced towards, 49 to 32), which gave an F1-score of approximately 0.75 per fold. We decided to run all three conditions to see if any of them had a good signature in distinguishing between HC's and SZ's, and we expected the condition where the button presses had resulting tones to give the best signature, because this would train the brain's suppression of its response to the expected stimuli more dramatically. **However, all three conditions did not give a signature which beat the control**.

For condition 1, the metrics worked out to:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond1_metrics.png)

The best fold was fold 0, and the rest of the folds matched the control exactly. The confusion matrices of fold 0 and fold 1 are, respectively:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond1_CM_fold0.png)

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond1_CM_fold1.png)

For condition 2, the metrics worked out to:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond2_metrics.png)

The best fold was fold 0, the worst fold was fold 2, and the rest of the folds matched the control exactly. The confusion matrices of fold 0 and fold 2 are, respectively:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond2_CM_fold0.png)

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond2_CM_fold2.png)

And finally, for condition 3, the metrics worked out to:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond3_metrics.png)

The wrost fold was fold 0, and the rest of the folds matched the control exactly. The confusion matrices of fold 0 and fold 1 are, respectively:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond3_CM_fold0.png)

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond3_CM_fold1.png)

# Conclusion and Future Work

From the results above, we can see that all three conditions (the button presses with a subsequent tones, the button presses without any tones emitted, and the tone played repeatedly without a button press) yielded no signature which distinguished between healthy controls and patients with schizophrenia. Using a random forest model, all three conditions could not beat the control of predicting that every test sample had schizophrenia.

The next obvious question is: why was there no discernible signature? Since we accounted for the imbalance in the data with a stratified K-fold cross validation and a balanced random forest, and we tuned the hyper-parameters, we propose the following explanations: the features generated are insufficient or cluttered, the chosen model is lacking predictive power, there is not enough training data, or the EEG test is simply not good enough to distinguish between health controls and patients with schizophrenia - ie, there is no signature present in this dataset.

It is possible that the features generated from `ts-fresh` are missing what is necessary to properly characterize an EEG, despite the sheer abundance of features generated. It is also possible that too many features are generated, and they combine with too much noise in the PCA. It would be worth doing a correlational analysis to see which features are most important to the response, or using a package or GitHub project which specifically generates features for EEG's.

In the future, I will also try to model with a linear regression or (convolutional) neural network, to see if that is more stable and predictive than a random forest. I would also hypothesize that using all 70 nodes might create too much imbalance or noise if a subset of those nodes are more important with respect to a possible signature. I will therefore build a new version of the project which trains the random forest on the nodes separately, then uses a majority vote to classify the sample that's scaled by the F1-score. A possible limitation to this method is that it would not characterize interaction between the nodes, so one could use an analysis to determine which nodes are most important, then combine those appropriately.

Since there were only 81 samples, 32 health controls and 49 patients with schizophrenia, it is conceivable that there is simply not enough data to train the model effectively. If possible, one could run a larger scale clinical trial to collect more data, but this would require a research proposal, funding, and the subsequent labour. **We suspect that this is the most likely source of a lack of signature in the model, and hesitate to conclude that there is no signature without more data**.

With that said, there are hints that a signature is not present in the EEG under these conditions. If one takes the 60 principal components and maps them down to 2 UMAP components using a UMAP embedding, we see no separation between health controls and patients with schizophrenia. For condition 1 fold 0, the UMAP is:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/cond1_fold0_UMAP_HC_SZ.png)

where blue are the health controls, and orange are the patients with schizophrenia, and this results generalizes to the two other conditions.

Although the preliminary results are disappointing so far, there is considerable potential that a new model can be built which finds a signature, and I am not prepared to make a hard conclusion on the hypothetical failure of this project without further analysis and modelling.