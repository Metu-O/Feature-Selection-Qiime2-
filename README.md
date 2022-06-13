# Feature-Selection-Qiime2

This repository describes the method used to test difference sci-kit learn feature selection methods for use in Qiime2 Naive Bayes taxonomy classifier

Note: Much of this code has been refactored from tax-credit-code and tax-credit-data (citations below)

# Feature selection methods tested

1. Filter Feature Selection
3. Wrapper Feature Selection
4. Embedded Feature Section 

# Filter Feature Selection
... add details here

# Wrapper Feature Selection
... add details here

# Embedded Feature Selection
... add details here 

# Initializing 
1. Clone tax-credit-code 

  '''
  git clone https://github.com/Metu-O/tax-credit-code)
  cd tax-credit-code
  pip install .
  '''

2. Clone tax-credit-data
.... add details here 

3. Clone Feature-Selection-Qiime2
.... add details here 

3. Move the tax-credit-data directory and Feature-Selection-Qiime2 codes into the tax-credit-code directory
.... add details here

4. Run all codes from inside the tax-credit-code directory

# Run codes 

Run feature selection python files (follow comments in codes for more information)
1. Naive_Bayes_Parameters.py contains code that runs the naive bayes classifier with no feature selection using qiime2 recommended parameters.
2. Univariate_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn filter feature selection method (SelectKBest).
3. Recursive_Feature_Selection_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn wrapper feature selection method (Recursive Feature Selection) ###not included
4. Feature_Selection_from_Model_Sweep.py contains code that runs the classifier while testing different parameters of a scikit-learn embedded feature selection method (SelectFromModel).

# Evaluate method accuracy

Compare method accuracy by running Evaluate_method_accuracy.py (follow comments in codes for more information)
