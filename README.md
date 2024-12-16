@author: jessicaguan
Created on Fri Dec 8 2023

This project creates 4 machine learning models (Naive Bayes and Random Forest) based on the [UCI dataset](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified) which contains data on individuals' personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism, impulsiveness, and sensation seeking) and their drug consumption habits.
The dataset contains informaion on several drug habits, but this project focuses on cannabis and ketamine for brevity. The Naive Bayes and Random Forest models can also be further tuned and trained for other drugs included in the dataset. Project details and results can be found in the file `ML-poster.pdf`.

The data preprocessing was done in the Jupyter notebook, `ML-DataPreprocessing.ipynb`. SMOTE (Synthetic Minority Oversampling Technique) was applied for better results. Tuning was done in the files `naivebayes_tuning.m` and `randomforest_tuning.m`, while the training, validation and testing was done in `naivebayes_ket.m`, `naivebayes_cannabis.m`, `randomforest_cannabis.m`, `randomforest_ket.m`.

There are a total of 4 models:
- `nb_mdl_cannabis.mat`
- `nb_mdl_ketamine.mat`
- `rf_mdl_cannabis.mat`
- `rf_mdl_cannabis.mat`

`nb_mdl_cannabis.mat` and `nb_mdl_ketamine.mat` are Naive Bayes models that classifies someone as a user of cannabis or ketamine respectively, using the 7 personality traits as the features. 
`rf_mdl_cannabis.mat` and `rf_mdl_ketamine.mat` are Random Forest models that classifies someone as a user of cannabis or ketamine respectively, using the 7 personality traits as the features.

Data used:
- `xTestNBCannabis` (Features for Naive Bayes model to classify individuals as non-users or users of cannabis)
- `xTestNBKetamine` (Features for Naive Bayes model to classify individuals as non-users or users of ketamine)
- `xTestRFCannabis` (Features for Random Forest model to classify individuals as non-users or users of cannabis)
- `xTestRFKetamine` (Features for Random Forest model to classify individuals as non-users or users of ketamine)
- `yTestNBCannabis` (y outputs for cannabis Naive Bayes model)
- `yTestNBKetamine` (y outputs for ketamine Naive Bayes model)
- `yTestRFCannabis` (y outputs for cannabis Random Forest model)
- `yTestRFKetamine` (y outputs for ketamine Random Forest model)
- 
For reproducibility:
- To reproduce the testing results for `nb_mdl_cannabis.mat`, load the model in the command window with the following command: load('nb_mdl_cannabis.mat). Then, run the code.
- To reproduce the testing results for `nb_mdl_ketamine.mat`, load the model in the command window with the following command: load('nb_mdl_ketamine.mat). Then, run the code.
- To reproduce the testing results for `rf_mdl_cannabis.mat`, load the model in the command window with the following command: load('rf_mdl_cannabis.mat). Then, run the code.
- To reproduce the testing results for `rf_mdl_ketamine.mat`, load the model in the command window with the following command: load('rf_mdl_ketamine.mat). Then, run the code.
