#!/usr/bin/env python
# coding: utf-8

# Run this notebook in the qiime2-2022.2 environment

# import modules

# In[2]:


import os
from os.path import join, exists, split, sep, expandvars 
from os import makedirs, getpid
from glob import glob
from shutil import rmtree
import csv
import json
import tempfile
from itertools import product

from qiime2.plugins import feature_classifier
from qiime2 import Artifact
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.naive_bayes import MultinomialNB
from q2_feature_classifier.classifier import spec_from_pipeline
from sklearn.feature_extraction.text import HashingVectorizer

from q2_types.feature_data import DNAIterator
from qiime2.plugin import Int, Str, Float, Bool, Choices, Range
from q2_types.feature_table import FeatureTable, RelativeFrequency
from q2_types.feature_data._transformer import (_read_from_fasta, _taxonomy_formats_to_dataframe, 
                                                _fastaformats_to_series)
from q2_feature_classifier._skl import _extract_reads, _specific_fitters, fit_pipeline
from q2_types.feature_data import (TSVTaxonomyFormat, HeaderlessTSVTaxonomyFormat, 
                                   FeatureData, Taxonomy, Sequence, DNAIterator, 
                                   DNAFASTAFormat)
from q2_feature_classifier.classifier import _load_class, spec_from_pipeline, pipeline_from_spec
from pandas import DataFrame

from tax_credit.framework_functions import (
    gen_param_sweep, generate_per_method_biom_tables, move_results_to_repository)


# Environment preparation 

# In[4]:


## project_dir should be the directory where you've downloaded (or cloned) the 
## tax-credit-data repository. 
project_dir = expandvars('$HOME/tax-credit-data/')
analysis_name = 'mock-community'
data_dir = join(project_dir, 'data', analysis_name)
precomputed_dir = join(project_dir, 'data', 'precomputed-results', analysis_name)
results_dir = join(project_dir, 'temp_results_narrow')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
reference_database_dir = join(project_dir, 'data','ref_dbs')


# In[ ]:


#sanity check
project_dir


# Utility Methods
# #The below methods are used to load the data, prepare the data, parse the classifier 
# #and classification parameters, and fit and run the classifier.

# In[6]:


def train_and_run_classifier(method_parameters_combinations, reference_dbs,
                             pipelines, sweep, verbose=False, n_jobs=4):
    '''Train and run q2-feature-classifier across a parameter sweep.
    method_parameters_combinations: dict of dicts of lists
        Classifier methods to run and their parameters/values to sweep
        Format: {method_name: {'parameter_name': [parameter_values]}}
    reference_dbs: dict of tuples
        Reference databases to use for classifier training.
        Format: {database_name: (ref_seqs, ref_taxonomy)}
    pipelines: dict
        Classifier pipelines to use for training each method.
        Format: {method_name: sklearn.pipeline.Pipeline}
    sweep: list of tuples
        output of gen_param_sweep(), format:
        (parameter_output_dir, input_dir, reference_seqs, reference_tax, method, params)
    n_jobs: number of jobs to run in parallel.
    '''
    # train classifier once for each pipeline param combo
    for method, db, pipeline_param, subsweep in generate_pipeline_sweep(
            method_parameters_combinations, reference_dbs, sweep):
        ref_reads, ref_taxa = reference_dbs[db]
        # train classifier
        classifier = train_classifier(
            ref_reads, ref_taxa, pipeline_param, pipelines[method], verbose=verbose)
        # run classifier. Only run in parallel once classifier is trained,
        # to minimize memory usage (don't want to train large refs in parallel)
        Parallel(n_jobs=n_jobs)(delayed(run_classifier)(
            classifier, output_dir, input_dir, split_params(params)[0], verbose=verbose)
            for output_dir, input_dir, rs, rt, mt, params in subsweep)

            
def generate_pipeline_sweep(method_parameters_combinations, reference_dbs, sweep):
    '''Generate pipeline parameters for each classifier training step'''
    # iterate over parameters
    for method, params in method_parameters_combinations.items():
        # split out pipeline parameters
        classifier_params, pipeline_params = split_params(params)
        # iterate over reference dbs
        for db, refs in reference_dbs.items():
            # iterate over all pipeline parameter combinations
            for param_product in product(*[params[id_] for id_ in pipeline_params]):
                # yield parameter combinations to use for a each classifier
                pipeline_param = dict(zip(pipeline_params, param_product))
                subsweep = [p for p in sweep if split_params(p[5])[1] 
                            == pipeline_param and p[2] == refs[0]]
                yield method, db, pipeline_param, subsweep


def train_classifier(ref_reads, ref_taxa, params, pipeline, verbose=False):
    ref_reads = Artifact.load(ref_reads)
    ref_taxa = Artifact.load(ref_taxa)
    pipeline.set_params(**params)
    spec = json.dumps(spec_from_pipeline(pipeline))
    if verbose:
        print(spec)
    classifier = feature_classifier.methods.fit_classifier_sklearn(ref_reads, ref_taxa, spec)
    return classifier.classifier


def run_classifier(classifier, output_dir, input_dir, params, verbose=False):    
    # Classify the sequences
    rep_seqs = Artifact.load(join(input_dir, 'rep_seqs.qza'))
    if verbose:
        print(output_dir)
    classification = feature_classifier.methods.classify_sklearn(rep_seqs, classifier, **params)
    
    # Save the results
    makedirs(output_dir, exist_ok=True)
    output_file = join(output_dir, 'taxonomy.tsv')
    dataframe = classification.classification.view(DataFrame)
    dataframe.to_csv(output_file, sep='\t', header=False)

    
def split_params(params):
    classifier_params = feature_classifier.methods.                        classify_sklearn.signature.parameters.keys()
    pipeline_params = {k:v for k, v in params.items()
                        if k not in classifier_params}
    classifier_params = {k:v for k, v in params.items() 
                         if k in classifier_params}
    return classifier_params, pipeline_params


# Preparing the method/parameter combinations and generating commands
# #Now we set the methods and method-specific parameters that we want to sweep. 
# #Modify to sweep other methods.

# In[8]:


dataset_reference_combinations = [
 ('mock-1', 'gg_13_8_otus'), 
 ('mock-2', 'gg_13_8_otus'), 
 ('mock-3', 'gg_13_8_otus'), 
 ('mock-4', 'gg_13_8_otus'), 
 ('mock-5', 'gg_13_8_otus'), 
 ('mock-6', 'gg_13_8_otus'), 
 ('mock-7', 'gg_13_8_otus'), 
 ('mock-8', 'gg_13_8_otus'),
 ('mock-12', 'gg_13_8_otus'),
 ('mock-13', 'gg_13_8_otus'),
 ('mock-14', 'gg_13_8_otus'),
 ('mock-15', 'gg_13_8_otus'),
 ('mock-16', 'gg_13_8_otus'),
 ('mock-18', 'gg_13_8_otus'),
 ('mock-19', 'gg_13_8_otus'),
 ('mock-20', 'gg_13_8_otus'),
 ('mock-21', 'gg_13_8_otus'),
 ('mock-22', 'gg_13_8_otus'),   
]

method_parameters_combinations = {
    'q2-NB': {'confidence': [0.7],
                         'classify__alpha': [0.001],
                         'feat_ext__ngram_range': [[8,8]]}}
    
reference_dbs = {'gg_13_8_otus' : (join(reference_database_dir, 'gg_13_8_otus/99_otus_v4.qza'), 
                                   join(reference_database_dir, 'gg_13_8_otus/99_otu_taxonomy_clean.tsv.qza'))}


# Preparing the pipelines
# #The below pipelines are used to specify the scikit-learn classifiers that are used for assignment.

# In[10]:


# pipeline params common to all classifiers are set here
hash_params = dict(
    analyzer='char_wb', n_features=8192, ngram_range=[8, 8],alternate_sign=False)


# any params common to all classifiers can be set here
classify_params = dict()


def build_pipeline(classifier, hash_params, classify_params):
    return Pipeline([
        ('feat_ext', HashingVectorizer(**hash_params)),
        ('classify', classifier(**classify_params))])
    
# Now fit the pipelines.
pipelines = {'q2-NB': build_pipeline(
                 MultinomialNB, hash_params, classify_params)}
pipelines


# Test - to test subset of dataset

# In[12]:


#dataset_reference_combinations = [
# ('mock-1', 'gg_13_8_otus'), # formerly S16S-1
#]
#
#method_parameters_combinations = {
 #   'q2-NB': {'confidence': [0.7],
  #                       'classify__alpha': [0.001],
   #                      'feat_ext__ngram_range': [[8,8]],
    #                     'feat_sel__estimator': [MultinomialNB()]}}
#    
#reference_dbs = {'gg_13_8_otus' : (join(reference_database_dir, 'gg_13_8_otus/99_otus_v4.qza'), 
#                                   join(reference_database_dir, 'gg_13_8_otus/99_otu_taxonomy_clean.tsv.qza'))}


# Do the Sweep

# In[14]:


sweep = gen_param_sweep(data_dir, results_dir, reference_dbs,
                        dataset_reference_combinations,
                        method_parameters_combinations)
sweep = list(sweep)


# A quick sanity check never hurt anyone...

# In[16]:


print(len(sweep))
sweep[0]


# In[17]:


train_and_run_classifier(method_parameters_combinations, reference_dbs, pipelines, sweep, verbose=True, n_jobs=1)


# In[18]:


#Generate per-method biom tables
#Modify the taxonomy_glob below to point to the taxonomy assignments that were generated above. 
#This may be necessary if filepaths were altered in the preceding cells.


# In[19]:


taxonomy_glob = join(results_dir, 'mock-*', 'gg_13_8_otus', 'q2-NB', '*', 'taxonomy.tsv')
generate_per_method_biom_tables(taxonomy_glob, data_dir)


# In[20]:


#Move result files to repository
#Add results to the short-read-taxa-assignment directory 
#(e.g., to push these results to the repository or compare with other precomputed results 
#in downstream analysis steps). The precomputed_results_dir path and methods_dirs glob below should not need to be changed unless if substantial changes were made to filepaths in the preceding cells.


# In[21]:


precomputed_results_dir = join(project_dir, "data", "precomputed-results", analysis_name)
method_dirs = glob(join(results_dir, '*', '*', '*', '*'))
move_results_to_repository(method_dirs, precomputed_results_dir)

