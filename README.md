(incomplete)

# Goal

The aim of this project is to predict whether or not a patient has schizophrenia based on their electroencephalogram (EEG) reading using a machine learning model.

# Motivation

Schizophrenia is a devastating mental health disorder which affects approximately [0.7% of the population](https://academic.oup.com/epirev/article/30/1/67/621138?login=false), meaning there are an estimated 56 million people with schizophrenia worldwide. This disorder can lead to [significant disability](https://pubmed.ncbi.nlm.nih.gov/11153962/), and an [estimated 6%](https://www.neomed.edu/medicine/wp-content/uploads/sites/2/Suicide-Risk-in-First-Episode-Psychosis-A-Selective-Review-of-Current-Literature.pdf) of people with schizophrenia will commit suicide in their lifetime - a 6x higher rate than the [US national average](https://afsp.org/suicide-statistics/) - with [22% attempting suicide](https://www.sciencedirect.com/science/article/abs/pii/S0920996409004964?via%3Dihub) within 7.4 years of the first episode of psychosis.

Treatment options are limited to antipsychotic medications, which carry a high burden of [side effects](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3384511/), and varying effectivity across the patient population. Nevertheless, access to treatment in a timely manner is a crucial step in the early intervention of psychotic-type illnesses, since effective treatment regimes can help a patient lead a fulfilling life.

However, a diagnosis of schizophrenia can only be made by a psychiatrist, and patients typically go [74 weeks](https://ajp.psychiatryonline.org/doi/full/10.1176/appi.ajp.2015.15050632) (1.4 years) from onset of initial symptoms to treatment with a perscribed antipsychotic. This is in stark contrast to a disease like cancer, which typically takes several weeks to diagnose. The difference is that schizophrenia currently has no physiological test - that is, there is no "litmus test" for schizophrenia, or for any known mental disorder. Establishing such tests is a major objective in the modern treatment of mental illness.

Access to a cost effective, non-invasive, rapid test for the diagnosis of patients with schizophrenia would be a major breakthrough in the field. An EEG, which records electrical activity of the brain through nodes on the head, could provide a viable candidate for a testing procedure. The average cost of an EEG reading in the United States is [$200](https://www.researchgate.net/profile/Richard-Lipton/publication/230131715_Cost_of_Health_Care_Among_Patients_With_Chronic_and_Episodic_Migraine_in_Canada_and_the_USA_Results_From_the_International_Burden_of_Migraine_Study_IBMS/links/5e0a72224585159aa4a6eee2/Cost-of-Health-Care-Among-Patients-With-Chronic-and-Episodic-Migraine-in-Canada-and-the-USA-Results-From-the-International-Burden-of-Migraine-Study-IBMS.pdf), the procedure involves the non-invasive placement of painless electrodes on the head, and software like a machine learning algorithm can provide a rapid evaluation of an EEG.

# Solution

Here, we propose that an EEG reading provides enough information to predict whether or not a patient has schizophrenia within a high degree of accuracy. This would act as a useful augmentation strategy for a psychiatrist's diagnostic procedure. To accomplish this, we will build a machine learning model on EEG data from a population of patients with and without schizophrenia.

# Data

[insert: outline with links to Kaggle]

# Code

All the software is written in Python scripts (located in `epg/`) and Jupyter notebooks (located in `notebooks/`). Some noteable packages used are: `ts-fresh`, `scikit-learn`, and `pytorch`.

### Running the Code

[insert: description]

# Model

[insert: stuff]

### Results

[insert: stuff]

### Future Work

[insert: stuff]