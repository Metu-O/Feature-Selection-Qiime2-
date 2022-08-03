# Feature-Selection-Qiime2

This repository describes the methods used to test different sci-kit learn feature selection methods as part of Qiime2 q2-classifier.

# Embedded Feature Selection

# 1. SelectFromModel Using Random Forest Estimator

The sklearn random forest classifier (RFC) was used as a base estimator in the feature selection step of my classification pipeline. RFC has a feature_importances_ attribute after fitting, which was used as the importance_getter parameter. For more details on sklearn's SelectFromModel, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html). For more details about sklearn's RFC, click on this other [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

# 2. SelectFromModel Using Stochastic Gradient Descent Estimator

The sklearn stochastic gradient descent (SGD) classifier was used as a base estimator in the feature selection step of my classification pipeline. RFC has a coef_ attribute after fitting, which was used as the importance_getter parameter. For more details about sklearn's SGDClassifier, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html). 

# 3. SelectFromModel Using Multinomial Naive Bayes Estimator

The sklearn multinomial naive bayes (MultinomialNB) classifier was used as a base estimator in the feature selection step of my classification pipeline. RFC has a feature_log_prob_ attribute after fitting, which was used as the importance_getter parameter. For more details about sklearn's MultinomialNB, click on this [link](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html). 

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
2. SelectFromModel_MultinomialNB.py contains code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the MultinomialNB estimator. 
3. SelectFromModel_RandomForest.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the RandomForestClassifier estimator. 
4. SelectFromModel_SDG.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the stochastic gradient descent (SDG) estimator. 

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)

# Citation

1. Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z

2. Osele, M. Machine Learning for Biological Data. ... (incomplete). 

3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
