"""
Created on Fri Dec  8 18:21:34 2023

@author: jessicaguan
"""

There are a total of 4 models, nb_mdl_cannabis.mat, nb_mdl_ketamine.mat, rf_mdl_cannabis.mat, and rf_mdl_cannabis.mat. nb_mdl_cannabis.mat and nb_mdl_ketamine.mat are naive bayes models that classifies someone as a user of cannabis or ketamine respectively, based on the 7 personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism, impulsiveness, and sensation seeking). rf_mdl_cannabis.mat and rf_mdl_ketamine.mat are random forest models that classifies someone as a user of cannabis or ketamine respectively, using the 7 personality traits as the features.

To reproduce the testing results for nb_mdl_cannabis.mat, load the model in the command window with the following command: load('nb_mdl_cannabis.mat). Then, run the code.

To reproduce the testing results for nb_mdl_ketamine.mat, load the model in the command window with the following command: load('nb_mdl_ketamine.mat). Then, run the code.

To reproduce the testing results for rf_mdl_cannabis.mat, load the model in the command window with the following command: load('rf_mdl_cannabis.mat). Then, run the code.

To reproduce the testing results for rf_mdl_ketamine.mat, load the model in the command window with the following command: load('rf_mdl_ketamine.mat). Then, run the code.