# Feature-Selection-Qiime2

This repository describes the methods used to test different sci-kit learn feature selection methods as part of Qiime2 q2-classifier.

Note: Much of the code and data in the tax_credit folder has been refactored from tax-credit-code and tax-credit-data (citations below)

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
1. Clone tax-credit-code 

  ```
  git clone https://github.com/Metu-O/tax-credit-code
  cd tax-credit-code
  pip install .
  ```

2. Clone tax-credit-data

 ```
  git clone https://github.com/Metu-O/tax-credit-data
  pip install .
  ```
  
3. Clone Feature-Selection-Qiime2

  ```
  git clone https://github.com/Metu-O/Feature-Selection-Qiime2
  pip install .
  ```

4. tax-credit-data and Feature-Selection-Qiime2 should be cloned into the tax-credit-code directory

# Run codes 

Run feature selection python files (follow comments in codes for more information)
1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 recommended parameters.
2. Univariate_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn filter feature selection method (SelectKBest).
3. Recursive_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn wrapper feature selection method (Recursive Feature Selection) ###not included
4. Feature_Selection_from_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn embedded feature selection method (SelectFromModel).

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)
