#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30Gb
#SBATCH --output=metu_trial.out
#SBATCH --time=60:00:00


source /opt/conda/etc/profile.d/conda.sh
conda activate /home/SE/BMIG-6202-MSR/qiime2-2022.2

cp -r ~/Feature-Selection-Qiime2 /scratch/metu/input/
cp -r ~/tax-credit-data /scratch/metu/input/
cp ~/Feature-Selection-Qiime2/Naive_Bayes_Parameters.py /scratch/metu/input
cp ~/Feature-Selection-Qiime2/SelectFromModel_MultinomialNB.py /scratch/metu/input
cp ~/Feature-Selection-Qiime2/SelectFromModel_RandomForest.py /scratch/metu/input
cp ~/Feature-Selection-Qiime2/SelectFromModel_SDG.py /scratch/metu/input
cp ~/Feature-Selection-Qiime2/Evaluate_Method_Accuracy.py /scratch/metu/input

cd /scratch/metu/input

python Naive_Bayes_Parameters.py \
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
python SelectFromModel_MultinomialNB.py \
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
python SelectFromModel_RandomForest.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
python SelectFromModel_SDG.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
ipython Evaluate_Method_Accuracy.py\
  -r project_dir \
  -e expected_results_dir \
  -m mock_results_fp  \
  -s results_dirs\
  -o mock_dir\
  -i min_count\
  -t taxonomy_level_range\
  -u outdir
  
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 
echo "Script completed in $(format_time $SECONDS)"
