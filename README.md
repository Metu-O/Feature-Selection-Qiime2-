# Feature-Selection-Qiime2

This repository describes the methods used to test different sci-kit learn feature selection methods as part of Qiime2 q2-classifier.

# Feature selection methods tested

1. Filter Method - Univariate Feature Selection
3. Wrapper Method - Recursive Feature Selection
4. Embedded Method - Tree-based Feature Selection with SelectFromModel
# Filter Feature Selection
... add details here

# Wrapper Feature Selection
... add details here

# Embedded Feature Selection
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
1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 recommended parameters.
2. Univariate_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn filter feature selection method (SelectKBest).
3. Recursive_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn wrapper feature selection method (Recursive Feature Selection) ###not included
4. Feature_Selection_from_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn embedded feature selection method (SelectFromModel).

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)

# Citation

Bokulich, N.A., Kaehler, B.D., Rideout, J.R. et al. Optimizing taxonomic classification of marker-gene amplicon sequences with QIIME 2’s q2-feature-classifier plugin. Microbiome 6, 90 (2018). https://doi.org/10.1186/s40168-018-0470-z
