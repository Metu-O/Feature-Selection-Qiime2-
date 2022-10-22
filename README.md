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
# ALL CODES MUST BE RUN IN A QIIME2 ENVIRONMENT (see this [link](https://docs.qiime2.org/2022.8/install/) on how to install the QIIME 2 Core 2022.8 distribution).

Run the feature selection python files in the 'Feature-Selection-Qiime2' directory 

1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 q2-classifier recommended parameters. Naive_Bayes_Parameters.py allows user to run codes with defaults or user input. I strongly advise using defaults except you are adept with directories. Run "help" to see usage.
```
python Naive_Bayes_Parameters.py -h
```
Run script with defaults 

```
python Naive_Bayes_Parameters.py 
```
Run script with user input 
```
python Naive_Bayes_Parameters.py \
    --tax_credit_data_path="your own tax-credit-data path" \
    --reference_database="you own reference_database dict" 
```
  
2. SelectFromModel_MultinomialNB.py contains code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the MultinomialNB estimator. Run "help" to see usage.
```
python SelectFromModel_MultinomialNB.py -h
```
Run script with defaults
```
python SelectFromModel_MultinomialNB.py 
```
Run script with user input 
```
python SelectFromModel_MultinomialNB.py \
  --tax_credit_data_path="your own tax-credit-data path" \
  --reference_database="you own reference_database dict" 
```

3. SelectFromModel_RandomForest.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the RandomForestClassifier estimator. Run "help" to see usage.
```
python SelectFromModel_RandomForest.py -h
```
Run script with defaults
```
python SelectFromModel_RandomForest.py
```
Run script with user input
```
python SelectFromModel_RandomForest.py\
  --tax_credit_data_path="your own tax-credit-data path" \
   --reference_database="you own reference_database dict" 
```

7. SelectFromModel_SDG.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the stochastic gradient descent (SDG) estimator. Run "help" to see usage.
```
python SelectFromModel_SDG.py -h
```
Run script with defaults
```
python SelectFromModel_SDG.py
```
Run script with user input
```
python SelectFromModel_SDG.py\
  --tax_credit_data_path="your own tax-credit-data path" \
  --reference_database="you own reference_database dict" 
```

Note: running these codes takes hours to run and may require a high computing processor. Do not wait around.

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)

Evaluate_method_accuracy.ipynb is the notebook version of Evaluate_method_accuracy.py, showing all plots and figures. 

```
python Evaluate_Method_Accuracy.py\
  -t tax_credit_data_path \
  -e expected_results_dir \
  -m mock_results_fp  \
  -s results_dirs\
  -o mock_dir\
  -u outdir
```

# Bash Script runs all codes 

Note: The bash script was personalized to my personal environments and filepaths. Edit before use.

# Conclusion

Feature-Selection-Qiime2 performed better than regular qiime2 more than 70% of the time. To find out more about the metrics used for comparison, read my thesis (will be cited once published). 

# Citation

1. Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z

2. Osele, M. Machine Learning for Biological Data. ... (not yet available). 

3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
