# Feature-Selection-Qiime2

This repository describes the methods used to test different sci-kit learn feature selection methods as part of Qiime2 q2-classifier.

# Feature selection methods tested

1. Filter Method - Univariate Feature Selection
2. Embedded Method - Using SelectFromModel

# Univariate Feature Selection Methods Tested
... add details here

# SelectFromModel Feature Selection Methods Tested
... add details here 

# Initializing 

1. Clone tax-credit-data

 ```
  git clone https://github.com/Metu-O/tax-credit-data
  ```
  
2. Clone Feature-Selection-Qiime2

  ```
  git clone https://github.com/Metu-O/Feature-Selection-Qiime2
  ```


# Run codes 

Run feature selection python files (follow comments in codes for more information)
1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 q2-classifier recommended parameters.
2. Univariate_Feature_Selection_Chi2.py contains code that runs the classifier with univariate feature selection, specifically sklearn's chi2 method.   
3. Univariate_Feature_Selection_F_Classif.py contains code that runs the classifier with univariate feature selection, specifically sklearn's f_classif method.
4. Univariate_Feature_Selection_MI_Classif.py contains code that runs the classifier with univariate feature selection, specifically sklearn's mutual_info_classif method.
5. SelectFromModel_MultinomialNB.py contains code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the MultinomialNB estimator. 
6. SelectFromModel_RandomForest.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the RandomForestClassifier estimator. 
7. SelectFromModel_SDG.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the stochastic gradient descent (SDG) estimator. 

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)

# Citation

Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z
