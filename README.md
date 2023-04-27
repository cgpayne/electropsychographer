# Introduction

### Goal

The aim of this project is to predict whether or not a patient has schizophrenia based on their electroencephalogram (EEG) reading using a machine learning model.

### Motivation

Schizophrenia is a devastating mental health disorder which affects approximately [0.7% of the population](https://academic.oup.com/epirev/article/30/1/67/621138?login=false), meaning there are an estimated 56 million people with schizophrenia worldwide. This disorder can lead to [significant disability](https://pubmed.ncbi.nlm.nih.gov/11153962/), and an [estimated 6%](https://www.neomed.edu/medicine/wp-content/uploads/sites/2/Suicide-Risk-in-First-Episode-Psychosis-A-Selective-Review-of-Current-Literature.pdf) of people with schizophrenia will commit suicide in their lifetime - a 6x higher rate than the [US national average](https://afsp.org/suicide-statistics/) - with [22% attempting suicide](https://www.sciencedirect.com/science/article/abs/pii/S0920996409004964?via%3Dihub) within 7.4 years of the first episode of psychosis.

Treatment options are limited to antipsychotic medications, which carry a high burden of [side effects](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3384511/), and varying effectivity across the patient population. Nevertheless, access to treatment in a timely manner is a crucial step in the early intervention of psychotic-type illnesses, since effective treatment regimes can help a patient lead a fulfilling life.

However, a diagnosis of schizophrenia can only be made by a psychiatrist, and patients typically go [74 weeks](https://ajp.psychiatryonline.org/doi/full/10.1176/appi.ajp.2015.15050632) (1.4 years) from the onset of initial symptoms to treatment with a prescribed antipsychotic. This is in stark contrast to a disease like cancer, which typically takes several weeks to diagnose. The difference is that schizophrenia currently has no physiological test - that is, there is no "litmus test" for schizophrenia, or for any known mental disorder. Establishing such tests is a major objective in the modern treatment of mental illness.

Access to a cost effective, non-invasive, rapid test for the diagnosis of patients with schizophrenia would be a major breakthrough in the field. An EEG, which records electrical activity of the brain through electrodes on the head, could provide a viable candidate for a testing procedure. The average cost of an EEG reading in the United States is [\$200](https://www.researchgate.net/profile/Richard-Lipton/publication/230131715_Cost_of_Health_Care_Among_Patients_With_Chronic_and_Episodic_Migraine_in_Canada_and_the_USA_Results_From_the_International_Burden_of_Migraine_Study_IBMS/links/5e0a72224585159aa4a6eee2/Cost-of-Health-Care-Among-Patients-With-Chronic-and-Episodic-Migraine-in-Canada-and-the-USA-Results-From-the-International-Burden-of-Migraine-Study-IBMS.pdf), the procedure involves the non-invasive placement of painless electrodes on the head, and software like a machine learning algorithm can provide a rapid evaluation of an EEG.

### Solution

Here, I propose that an EEG reading provides enough information to predict whether or not a patient has schizophrenia within a high degree of accuracy. This would act as a useful augmentation strategy for a psychiatrist's diagnostic procedure. To accomplish this, I will build a machine learning model on EEG data from a population of patients with and without schizophrenia.

# Data <a id="data"></a>

I retrieved data from a [study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4059422/) of healthy controls (HC) versus patients with schizophrenia (SZ), from a dataset in two parts on Kaggle [here](https://www.kaggle.com/datasets/broach/button-tone-sz?resource=download) and [here](https://www.kaggle.com/datasets/broach/buttontonesz2). There were 32 HC's and 49 SZ's, for a total of 81 samples.

In this study, it is described that HC's have a mechanism in the brain which suppresses neural activity from expected or internal stimuli. For instance, if one pushes a button that emits a tone, and that tone is expected by the brain, the brain will suppress neural activity associated with the auditory stimulus of the tone. This allows for more efficient functioning in the sensory cortex. However, patients that have schizophrenia are known to have less suppression of this neural activity, meaning their brains treat external and internal stimuli in a similar manner.

It is posited that this dysfunction in neural suppression contributes to the experience of sensory hallucinations in patients with schizophrenia. Furthermore, the authors expect that one can characterize this neural activity on an EEG reading through 70 electrodes on the patient's head which read the electrical activity of the brain.

I therefore asked the question: **is there enough of a signature in these EEG readings to distinguish between a healthy control and a patient with schizophrenia?** If so, can a machine learning model be trained on this difference, and then predict whether or not a new patient has schizophrenia based on their EEG? A model like this could be a major tool to psychiatrists as a physiological test (of which none currently exists) for schizophrenia.

In the study, a button was repeatedly pressed which emitted a tone and the EEG read the brain's electrical activity as it reacted to the stimulus of the noise. In a HC, as the button is pressed in succession, the brain should expect the tone and suppress the related neural activity, whereas a SZ's brain will not. There were three conditions tested: the button press with a subsequent tone, the button press without a tone (silence), and simply a played tone without a button press involved. Each condition had an intended 100 trials (ie, 100 presses/tones), and each trial had 3072 time points measured on the EEG, which gives us a time series for each of the 70 electrodes on the head. I chose to train on each condition separately to see which one had the best signature.

### Visualization <a id="visualization"></a>

To visualize the data, I will do two things: first I will normalize the time series using a min-max scaling to make the &mu;V amplitudes unitless, which I do to make effective comparisons; and then, without loss of generality, I will focus on the auditory cortex for context. The auditory cortex is split into two sections which run in parallel to each other across the skull: the primary auditory cortex and the secondary auditory cortex. Three electrodes run over those cortex sections, which I call the: front, middle, and back. Respectively, the electrodes on the left side of skull are labelled T7 (left-front), TP7 (left-middle), and P7 (left-back), whereas the on the right side they are labelled T8 (right-front), TP8 (right-middle), P8 (right-back).

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_electrodes_diagram.png)

Above, the 6 electrodes of interest are mapped across the skull, with the remaining 64 electrodes excluded for clarity. Below, we compare the three conditions for the HC (patient 1) on the top row and the SZ (patient 81) on the bottom row.

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_full/1vs2vs3_T7_T8_tight.png)

Here we can see that the HC and SZ are visibly different within both the T7 and T8 electrodes respectively. However, the difference between the conditions in each subplot is less clear. The most striking difference is in the right-front electrode T8 in the SZ patient, which suggests that there is asymmetry between the right and left hemispheres. It also corroborates the intuition that the HC should see less of a difference between the conditions due to the reduced neural response to the stimuli, whereas the SZ still measures a difference. In other words, the HC has similar neural activity when performing the press/tone vs press/silence task, but the SZ's brain still distinguishes between the conditions because its ability to suppress the response to the expected stimuli is impaired.

Now I plot each condition separately so we can compare the HC and SZ directly within each subplot for the P7 electrode.

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_full/HC_vs_SZ_cond123_P7.png)

The HC and SZ are comparable in condition 1 and condition 2, but more distinct in condition 3. In condition 3 we can clearly see wave patterns, these are manifestations of alpha, beta, and gamma brain waves.

Finally, we compare the left hemisphere (in the left column) with the right hemisphere (in the right column) from front to back for:

- condition 1:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_full/HC_vs_SZ_auditory_cortex_cond1_tight.png)

- condition 2:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_full/HC_vs_SZ_auditory_cortex_cond2_tight.png)

- condition 3:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/EEG_full/HC_vs_SZ_auditory_cortex_cond3_tight.png)

Again we see that condition 3 has the most variability, and that the hemispheres are asymmetric in SZ. Interestingly, in condition 1 and 2 for SZ, the left hemisphere's time series completely track with each other across the electrodes from front to back, but in the right hemisphere T8 and TP8 track, but P8 differs. For HC we can see symmetry between the hemispheres and that the electrodes track from front to back.

All of these visualizations show promise that there is a measurable difference between the EEG's of people with or without schizophrenia.

# Code

In this section I describe the structure of the code and how to run it in great detail. If you would like to see a summary of the model and the subsequent results, you can skip directly to [Model](#model) and [Results](#results) sections.

All the software is written in Python scripts (located in `epg/`) and Jupyter notebooks (located in `notebooks/models/`). Some notable packages used are: `ts-fresh`, `scikit-learn`, and `pytorch`. The Python version I used is `3.9.15` and there is a `requirements.txt` in the top directory.

There are two main phases of the software: the *script phase* and the *notebook phase*. In the *script phase* I handle the data processing in a production-ready manner, which involves reformatting the data, generating features from the normalized time series (using `ts-fresh`), removing outliers, creating stratified folds for cross validation, and finally running a PCA for dimensionality reduction. In the *notebook phase* I build the model in an exploratory manner, which uses the random forest algorithm that's optimized via a randomized search in a Jupyter notebook.

### Script Phase

This phase uses scripts under the `epg/` directory. After setting up the data (see [Data](#data)), I will follow these steps: data pruning, feature generation, outlier detection, splitting the data into folds, and finally a PCA.

##### User Config

To run the respective scripts, I use a user configuration file to set certain parameters about the run. To see an example of these parameters and how they relate to the individual scripts, refer to the `epg/config/example_config.py` file. In order to execute a script, one must copy this example into a file called `user_config.py` where the parameters can be toggled for the run.

`cp example_config.py user_config.py`

Note that `example_config.py` is tracked by git, whereas `user_config.py` is not, as seen in the `.gitignore` file.

##### Stages

I will now describe the individual scripts, their purpose, and how to run them. Each script can be found under the `epg/` directory, and they all make use of the `user_config.py`.

###### 1 - Data Pruning <a id="data-pruning"></a>

To reduce computational load, I work on only one trial out of the 100 trials per patient per condition. I assume that the later trials (which each represent a time series) will have the bigger difference in signature, since that's when the brain will be best trained against the tone in the button test. That is, patients without schizophrenia will now have a suppression of the neural activity from the predicted sound, whereas those with schizophrenia will not, in theory. So, I prune out the latest trial possible, where all three conditions are present, per patient. This restriction typically happens at trial 90, but sometimes lower or higher, but (for unexplained reasons) never at the final trial (ie, 100). I found the respective trials manually.

To run this, first set the `patients_dp` parameter in the `user_config.py`, which lists the patient dataset to be pruned and the trial to pull out. Then execute:

`python data_pruning.py`

The data is taken from the `archive_dir`, processed, and saved to the `pruned_dir` variables, which are set respectively in `epg/epglib/constants.py` file.

###### 2 - Feature Generation

I now use `ts-fresh` to generate a large range of features from the time series (which I first normalize using min-max scaling) via the `epg/feature_gen.py` script and then some post-processing using the `epg/feature_gen_post.py` script.

Set the config parameters, and then run:

`python feature_gen.py`

For the full dataset of all the patients, this will produce roughly 54,810 features from the normalized time series in under an hour using the `EfficientFCParameters` setting from `ts-fresh`. With more computational resources, one may also use the `ComprehensiveFCParameters` to produce an additional 420 features.

There are many features which have all zeros exactly and also approximately (see the `eps_flat` parameter in the `user_config.py`). I excise those in `epg/feature_gen_post.py`, and also excise any features where there would be all zeros in the training data (see the `test_size_fpg` parameter in the `user_config.py`). I also impute the data overall. Run:

`python feature_gen_post.py`

###### 3 - Outlier Removal

[insert: describe after updating the method]

###### 4 - Split the Data for Stratified K-fold CV

I now split the data into the desired number of folds for cross validation (or into a single standard test/training split if desired) using the `epg/split_the_data.py` script. I chose a stratified K-fold cross validation to retain the class imbalance present between the 32 healthy controls to 49 schizophrenia patients.

Set the `user_config.py` parameters (make sure the `Kfolds` parameter matches with the `test_size_fgp` parameters from the prior `feature_gen_post.py` run), and then run:

`python split_the_data.py kfold`

Note that you may use any other string in place of `kfold` to run a standard `test_train_split` as compared to the `StratifiedKFold` (both from `sklearn.model_selection`). The data is taken from the `fgen_dir`, processed, then saved to the `split_dir` (see the `epg/epglib/constants.py` file) in the form `X_train-*_<data_handle>.csv`, where the `*` wildcard refers to the fold number and the `<data_hanlde>` is set in the `user_config.py`.

###### 5 - PCA <a id="PCA"></a>

To reduce the ~10<sup>4</sup> features generated from the normalized time series to a more computationally efficient dimensionality, I use Principal Component Analysis (PCA). This will pay off when running models like random forest and alike, but I must make sure that I capture enough of the data's variance in a reasonable amount of principal components. To see the typical results of this PCA, refer to the [PCA](#pca-results) section - we can reduce the dimension to ~10. To run, try:

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

This is where the machine learning models are built and run. For a random forest using a stratified K-fold cross validation see the `random_forest_all.ipynb` notebook. In order to execute this notebook, all the data must be processed from [Stage 1](#data-pruning) through to [Stage 5](#PCA) in the *script phase* of the codebase. For the results from this notebook, see the [Results](#results) section.

The random forest model is built in the `OptimalRF` class, where I tune the hyper-parameters of the random forest (decision tree depth and forest size) using a randomized search procedure. I allowed for a possible 200 estimators and 20 levels of depth in the optimization. In the following cells I run the model, calculate all the associated metrics (like F1-score), and produce confusion matrices for each fold.

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

# Model <a id="model"></a>

In totality: the model used `ts-fresh` with `EfficientFCParamteres` to generate 54,810 features from the normalized EEG time series, per condition; then I used a PCA via `sklearn.decomposition`to reduce the dimensionality to about 60 principal components which capture 100% of the variance, per 5-fold for cross validation; and finally I used a random forest model `from sklearn.ensemble import RandomForestClassifier`, which I optimized using a random search, in `notebooks/models/random_forest_all.ipynb`.

### PCA <a id="pca-results"></a>

In a typical PCA, there was a reduction to around 60 principal components. For condition 2, for example, the cumulative explained variance went as follows:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/PCA_cond2/cummulative_explained_variance.png)

Somewhat ominously, there was no separation between healthy controls (HC) and patients with schizophrenia (SZ) in the PCA. However, we see that the first two principle components only capture 8.04% and 6.52% of the total variance:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/PCA_cond2/PC1_vs_PC2.png)

### Choice of Model

I then chose to start with a random forest model on the total ~60 principal components. As can be seen in the [Visualization](#visualization), many cyclic patterns arise in the EEG brain waves (alpha, beta, gamma) which occur in the 1ms to 1sec time scale across the ~3sec time series recorded. This would suggest that a convolutional neural network (CNN) would be useful, since it is good at distinguishing these sorts of patterns independent of shifts in time. However, a random forest is more robust and more efficient than a CNN for non-linear structured data with few samples. This is because random forests account for overfitting of the model via bagging, whereas neural nets have a tendency to overfit, hence they require larger datasets. In the future I will also train a logistic regression and a neural net for the purpose of comparison.

### Metrics

To assess the random forest model, I used a stratified K-fold (K = 5) cross validation, and averaged the following metrics for analysis: accuracy, precision, recall, and F1-score (the harmonic mean of the precision and recall).

In healthcare diagnostics, we care more about recall = true-positives / (true-positives + **false-negatives**) than we do about precision = true-positives / (true-positives + **false-positives**). This is because to maximize recall, we need low false-negatives, whereas to maximize precision, we need low false-positives; however, in diagnostics we would rather have no false-negatives (ie, we never predict someone doesn't have cancer when they actually do) and allow for some false-positives. In the case of a false-positive, physicians can use further tests to rule out a positive diagnosis, and so we prefer recall over precision.

However, as we will see in the results, my model will predict 100% recall in all cases. So to optimize the model, I will focus on the F1-score because the data is unbalanced.

# Results <a id="results"></a>

In this section we go over the preliminary results. For the unbalanced data, I used the `RandomForestClassifier(class_weight="balanced")` balanced classifier, and used a stratified 5-fold cross validation to keep the ratio of HC to SZ consistent. To optimize the hyper-parameters of the random forest (that is, the size of the random forest and the depth of the decision trees), I used the `RandomizedSearchCV` from `sklearn.model_selection`.

As a control, we want to beat a model which predicts all 1's (ie, SZ's, which the data was unbalanced towards, 49 to 32), which gives an F1-score of approximately 0.77. I decided to run all three conditions to see if any of them had a good signature in distinguishing between HC's and SZ's, and I expected the condition where the button presses had resulting tones to give the best signature, because this would train the brain's suppression of its response to the expected stimuli more dramatically. **However, all three conditions did not give a signature which beat the control**.

For condition 1, the metrics worked out to a recall of 90% and a F1-score of 0.72 (ie, the predictions don't beat the control):

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond1_metrics.png)

Note that the metrics are approximately the same between the control and the model. The confusion matrix of the predictions on the test set of condition 1 is:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond1_CM.png)

One can see that the model is almost always predicting 1's (SZ), and note that the 0 class is HC.

For condition 2, the metrics worked out to a recall of 100% and a F1-score of 0.77:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond2_metrics.png)

This means the predictions match the control exactly for condition 2, resulting in the following confusion matrix:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond2_CM.png)

And finally, for condition 3, the metrics also worked out to a recall of 100% and a F1-score of 0.77:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond3_metrics.png)

This means the predictions match the control exactly for condition 3, resulting in the following confusion matrix:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/cond3_CM.png)

# Conclusion and Future Work

From the results above, we can see that all three conditions (the button presses with a subsequent tones, the button presses without any tones emitted, and the tone played repeatedly without a button press) yielded no signature which distinguishes between healthy controls and patients with schizophrenia. Using a random forest model, all three conditions could not beat the control of predicting that every test sample had schizophrenia, with an F1-score of 0.77.

The next obvious question is: why was there no discernible signature? Since I accounted for the imbalance in the data with a stratified K-fold cross validation and a balanced random forest, and I tuned the hyper-parameters, I propose the following remaining explanations: the features generated are insufficient or cluttered, the chosen model is lacking predictive power, there is not enough training data, or the EEG test is simply not good enough to distinguish between healthy controls and patients with schizophrenia - ie, there is no signature present in this dataset.

It is possible that the features generated from `ts-fresh` are missing what is necessary to properly characterize an EEG, despite the sheer abundance of features generated. It is also possible that too many features are generated, and they combine with too much noise in the PCA. It would be worth doing a correlational analysis to see which features are most important to the response, or using a package or GitHub project which specifically generates features for EEG's.

In the future, I will also try to model with a logistic regression or a (convolutional) neural network, to see if they are more stable and predictive than a random forest. I would also hypothesize that using all 70 electrodes might create too much imbalance or noise if a subset of those electrodes are more important with respect to a possible signature. I will therefore build a new version of the project which trains the random forest on the electrodes separately, then use a majority vote (scaled by the F1-score) to classify the sample. A possible limitation to this method is that it would not characterize interaction between the electrodes, so one could use an analysis to determine which electrodes are most important, then combine those appropriately.

Since there were only 81 samples, 32 healthy controls and 49 patients with schizophrenia, it is conceivable that there is simply not enough data to train the model effectively. If possible, one could run a larger scale clinical trial to collect more data, but this would require a research proposal, funding, and the subsequent labour. **I suspect that this is the most likely source of a lack of signature in the model, and hesitate to conclude that there is no signature without more data**.

With that said, there are hints that a signature is not present in the EEG under these conditions. If one takes the total ~60 principal components and maps them down to 2 UMAP components using a UMAP embedding, we see no separation between healthy controls and patients with schizophrenia. For condition 1, the UMAP is:

![](https://raw.githubusercontent.com/cgpayne/electropsychographer/master/markdown_images/results/UMAP_cond1.png)

where blue are the healthy controls, and orange are the patients with schizophrenia - and these results generalizes to the two other conditions.

Although the preliminary results are disappointing so far, there is considerable potential that a new model can be built which finds a signature, and I am not prepared to make a hard conclusion on the acceptance of the null hypothesis of this project without further analysis and modelling.