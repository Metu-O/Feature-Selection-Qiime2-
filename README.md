# Feature-Selection-Qiime2

This repository outlines the methodologies employed to evaluate different scikit-learn feature selection methods within the Qiime2 q2-classifier framework.

# Embedded Feature Selection

# 1. SelectFromModel Using Multinomial Naive Bayes Estimator

The sklearn Multinomial Naive Bayes (MultinomialNB) classifier is utilized as the estimator for feature selection within the classification pipeline. The feature_log_prob_ attribute from RFC is employed as the importance_getter parameter. For more details about sklearn's MultinomialNB, refer to this [link](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

# 2. SelectFromModel Using Random Forest Estimator

In this approach, the sklearn Random Forest Classifier (RFC) serves as the estimator for feature selection within the classification pipeline. The feature_importances_ attribute from RFC is used as the importance_getter parameter. For further information on sklearn's SelectFromModel, visit this [link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html). Detailed information about sklearn's RFC is available [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). 

# 3. SelectFromModel Using Stochastic Gradient Descent Estimator

The sklearn Stochastic Gradient Descent (SGD) classifier is harnessed as the base estimator for feature selection within the classification pipeline. The coef_ attribute from RFC is used as the importance_getter parameter. Learn more about sklearn's SGDClassifier by referring to https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html.  

# Getting Started 

1. Clone the Repository

  ```
  git clone https://github.com/Metu-O/Feature-Selection-Qiime2
  ```
  
2. Navigate to the Repository

  ```
  cd Feature-Selection-Qiime2
  ```

3. Activate Qiime2 environment
   
   Follow the instructions provided in the [QIIME 2 installation guide](https://docs.qiime2.org/2022.8/install/) to activate the QIIME 2 Core 2022.8 distribution.

4. Run the scripts

   Execute the code using command lines or a bash script (customize the bash script according to your path). 

# Usage

1. **Naive Bayes Parameters Script**
   To run the Naive Bayes classifier with default parameters, use:
   
```
python Naive_Bayes_Parameters.py
```
  For custom settings:

```
python Naive_Bayes_Parameters.py -n 'database_name' -s 'path_to_sequences.qza' -t 'path_to_taxonomy.qza'
```
  Example: 
```
python Naive_Bayes_Parameters.py -n 'greengenes' -s '/path/to/sequences.qza' -t '/path/to/taxonomy.qza'
```



```

2. SelectFromModel_MultinomialNB.py contains code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the MultinomialNB estimator. Run "help" to see usage.
```
python SelectFromModel_MultinomialNB.py -h
```
Run script below to use defaults
```
python SelectFromModel_MultinomialNB.py 
```
Run script below with to change defaults with user input
```
python SelectFromModel_MultinomialNB.py \
   -n 'reference database name'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)' 
```

3. SelectFromModel_RandomForest.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the RandomForestClassifier estimator. Run "help" to see usage.
```
python SelectFromModel_RandomForest.py -h
```
Run script below to use defaults
```
python SelectFromModel_RandomForest.py
```
Run script below with to change defaults with user input
```
python SelectFromModel_RandomForest.py \
   -n 'reference database name e.g greengenes or SILVA'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)'
```

4. SelectFromModel_SDG.py code that runs the classifiers with a sklearn embedded feature selection method, SelectFromModel, using the stochastic gradient descent (SDG) estimator. Run "help" to see usage.
```
python SelectFromModel_SDG.py -h
```
Run script below to use defaults
```
python SelectFromModel_SDG.py
```
Run script below with to change defaults with user input
```
python SelectFromModel_SDG.py \
   -n 'reference database name'\
   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)' 
```

Compare method accuracy by running Evaluate_method_accuracy.py

NOTE: Evaluate_method_accuracy.ipynb is the notebook version of Evaluate_method_accuracy.py, showing all plots and figures. 

Run "help" to see usage.
```
python Evaluate_Method_Accuracy.py -h
```
Run script below to use defaults
```
python Evaluate_Method_Accuracy.py
```
Run script below with to change defaults with user input
```
python Evaluate_Method_Accuracy.py \
   -p 'save plots in this directory 
```

Note: Running these scripts can be time-consuming and resource-intensive.

# Bash Script (Optional)

The provided bash script can automate the execution of all codes. However, ensure you customize the script according to your environment and filepaths before use.

# Conclusion

The Feature-Selection-Qiime2 repository has demonstrated significant improvement over standard Qiime2 in more than 70% of cases. For further details on the evaluation metrics, refer to the thesis (to be cited upon publication). 

# Citation

1. Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z

2. Osele, M. Machine Learning for Biological Data. ... (not yet available). 

3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
