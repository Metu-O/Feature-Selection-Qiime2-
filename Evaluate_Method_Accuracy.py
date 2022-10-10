# Run this code in the qiime2-2022.2 environment.

# Import modules and packages
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
from os.path import join, exists, expandvars
import pandas as pd
import matplotlib.pyplot as plt
#from IPython.display import display, Markdown
import seaborn
import argparse
import pathlib
from tax_credit.plotting_functions import (pointplot_from_data_frame,
                                           boxplot_from_data_frame,
                                           heatmap_from_data_frame,
                                           per_level_kruskal_wallis,
                                           beta_diversity_pcoa,
                                           average_distance_boxplots,
                                           rank_optimized_method_performance_by_dataset)
from tax_credit.eval_framework import (evaluate_results,
                                       method_by_dataset_a1,
                                       parameter_comparisons,
                                       merge_expected_and_observed_tables,
                                       filter_df)

def evaluate_method_accuracy(project_dir,expected_results_dir, mock_results_fp,results_dirs,mock_dir,outdir):
#Experienced users can create their own path.  
    project_dir = expandvars('$HOME/tax-credit-data/')
    expected_results_dir = join(project_dir, "data/precomputed-results/", "mock-community")
    mock_results_fp = join(expected_results_dir, 'mock_results.tsv')
    results_dirs = [expected_results_dir]
    mock_dir = join(project_dir, "data", "mock-community")
    outdir = join(project_dir, "plots")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset_ids = ['mock-1', 'mock-2', 'mock-3', 'mock-4', 'mock-5','mock-6', 'mock-7', 'mock-8','mock-12','mock-13','mock-14', 'mock-15', 'mock-16','mock-18', 'mock-19', 'mock-20', 'mock-21', 'mock-22']
    method_ids = ['q2-NB', 'q2-SFM-RF', 'q2-SFM-SDG','q2-SFM-NB']
    ref_ids = ['gg_13_8_otus']
    # Find mock community pre-computed tables, expected tables, and "query" tables
    # Note: if you have added additional methods to add, set append=True. If you are attempting to recompute pre-computed results, set force=True.
    mock_results = evaluate_results(results_dirs, 
                                expected_results_dir, 
                                mock_results_fp, 
                                mock_dir,
                                taxonomy_level_range=range(2,7), #Define the range of taxonomic levels over which to compute accuracy scores. The                                   default will compute order (level 2) through species (level 6)
                                min_count=1, #Minimum number of times an OTU must be observed for it to be included in analyses. Edit this to analyze the effect of the minimum count on taxonomic results.'
                                taxa_to_keep=None, 
                                md_key='taxonomy', 
                                subsample=False,
                                per_seq_precision=True,
                                exclude=['other'],
                                dataset_ids=dataset_ids,
                                reference_ids=ref_ids,
                                method_ids=method_ids,
                                append=False,
                                force=True, #force=True the first time 
                                backup=False)
    #Compute and summarize precision, recall, and F-measure for mock communities
    color_palette={ 
    'q2-NB':'black',
    'q2-SFM-RF':'darkgreen',
    'q2-SFM-SDG':'red',
    'q2-SFM-NB':'blue',
    }
    y_vars = ["Precision", "Recall", "F-measure", "Taxon Accuracy Rate", "Taxon Detection Rate"]
    point = pointplot_from_data_frame(mock_results, "Level", y_vars, 
                                  group_by="Reference", color_by="Method",
                                  color_palette=color_palette)
    for k, v in point.items():
        v.savefig(join(outdir, 'mock-{0}-lineplots.pdf'.format(k)),bbox_inches = 'tight')

    # Heatmaps show the performance of individual method/parameter combinations at each taxonomic level, in each reference database
    heatmap_from_data_frame(mock_results, metric="Precision", rows=["Method", "Parameters"], cols=["Reference","Level"])
    plt.savefig(join(outdir, "Precision-heatmap.pdf"), bbox_inches = 'tight')
    
    heatmap_from_data_frame(mock_results, metric="Recall", rows=["Method", "Parameters"], cols=["Reference","Level"])
    plt.savefig(join(outdir, "Recall-heatmap.pdf"), bbox_inches = 'tight')
    
    heatmap_from_data_frame(mock_results, metric="F-measure", rows=["Method", "Parameters"], cols=["Reference","Level"])
    plt.savefig(join(outdir, "F-measure-heatmap.pdf"), bbox_inches = 'tight')
    
    heatmap_from_data_frame(mock_results, metric="Taxon Accuracy Rate", rows=["Method", "Parameters"], cols=    
                            ["Reference", "Level"])
    plt.savefig(join(outdir, "Taxon Accuracy Rate.pdf"), bbox_inches = 'tight')
    
    heatmap_from_data_frame(mock_results, metric="Taxon Detection Rate", rows=["Method", "Parameters"], cols=
                            ["Reference", "Level"])
    plt.savefig(join(outdir, "Taxon Detection Rate-heatmap.pdf"), bbox_inches = 'tight')

    #Now we will focus on results at species level (for genus level, change to level 5)
    # Method optimization
    # Which method/parameter configuration performed "best" for a given score? We can rank the top-performing configuration by dataset, method,    
    #and taxonomic level. 
    # First, the top-performing method/configuration combination by dataset.
    mock_results_6 = mock_results[mock_results['Level'] == 6]
    #pd.set_option('display.max_colwidth', None)
    for dataset in mock_results_6['Dataset'].unique():
        #display(Markdown('## {0}'.format(dataset)))
        best = method_by_dataset_a1(mock_results_6, dataset)
        #display(best)
        best.to_csv(join(outdir, '{0}-best_method.tsv'.format(dataset)),sep='\t')
    
    # Now we can determine which parameter configuration performed best for each method. Count best values in each     #column indicate how many   
    #samples a given method achieved within one mean absolute deviation of the best      
    #result (which is why they may sum to more than the total number of samples).
    for method in mock_results_6['Method'].unique():
        top_params = parameter_comparisons(
        mock_results_6, method, 
        metrics=['Taxon Accuracy Rate', 'Taxon Detection Rate', 'Precision', 'Recall', 'F-measure'])
        #display(Markdown('## {0}'.format(method)))
        #display(top_params[:5])
        top_params[:5].to_csv(join(outdir, '{0}-top_params.tsv'.format(method)),sep='\t')
        
    
    # Optimized method performance
    # And, finally, which method performed best at each individual taxonomic level for each reference dataset (i.e., for across all fungal and
    #bacterial mock communities combined)? 
    # For this analysis, we rank the top-performing method/parameter combination for each method at family through species levels. Methods are ranked 
    #by top F-measure, and the average value for each metric is shown (rather than count best as above). F-measure distributions are plotted for
    #each method, and compared using paired t-tests with FDR-corrected P-values. This cell does not need to be altered, unless if you wish to 
    #change the metric used for sorting best methods and for plotting.
    
    for metric in ["Taxon Accuracy Rate", "Taxon Detection Rate", "Precision", "Recall", "F-measure"]:
        #display(Markdown('## {0}'.format(metric)))
        boxes = rank_optimized_method_performance_by_dataset(mock_results,
                                                         dataset="Reference",
                                                         metric=metric,
                                                         level_range=range(6,7),
                                                         display_fields=["Method",
                                                                         "Parameters",
                                                                         "Taxon Accuracy Rate",
                                                                         "Taxon Detection Rate",
                                                                         "Precision",
                                                                         "Recall",
                                                                         "F-measure"],
                                                         paired=True,
                                                         parametric=True,
                                                         color=None,
                                                         color_palette=color_palette)
        for k, v in boxes.items():
            v.get_figure().savefig(join(outdir, 'mock-{0}-{1}-boxplots.pdf'.format(metric, k)), bbox_inches = 'tight')

def main():
    parser = argparse.ArgumentParser(
            description= 'The tax-credit-data directorsy contains all data used to prepare these codes. I am                   argparse to soft code important file paths needed to run this script. Experienced users can create                 their own paths')

    req = parser.add_argument_group('REQUIRED')
    req.add_argument('-r', '--project_dir', required=True, action='store',type=pathlib.Path,
             help='Project_dir should be tax-credit-data directory')
    req.add_argument('-e', '--expected_results_dir', required=True, action='store',type=pathlib.Path,
             help='expected_results_dir contains expected composition                                                                data in the structure expected_results_dir/<dataset name>/<reference name>/expected/')
    req.add_argument('-m', '--mock_results_fp', required=True, action='store',type=pathlib.Path,
             help='mock_results_fp designates the files to which summary results are written. If this                file exists, it can be read in to generate results plots, instead of computing new scores.')
    req.add_argument('-s', '--results_dirs', required=True, action='store',type=pathlib.Path,
             help='results_dirs should contain the directory or directories where results can be found.                              By default, this is the same location as expected results included with the project.                              If other results should be included, absolute paths to those directories should be                                added to this list.')
    req.add_argument('-o', '--mock_dir', required=True, action='store',type=pathlib.Path,
             help='directory containing mock community data, e.g., feature table without taxonomy')
    req.add_argument('-u', '--outdir', required=True, action='store',type=pathlib.Path,
             help='you can save plots in this directory')
    
    
    p = parser.parse_args()
    project_dir = p.project_dir
    expected_results_dir = p.expected_results_dir
    mock_results_fp  = p.mock_results_fp 
    results_dirs =p.results_dirs
    mock_dir = p.mock_dir
    outdir = p.outdir
    
    evaluate_method_accuracy(project_dir,expected_results_dir, mock_results_fp,results_dirs,mock_dir,outdir)

if __name__ == '__main__':
        main()
