#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30Gb
#SBATCH --output=metu_trial.out
#SBATCH --time=60:00:00

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 
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
  
echo "Naive_Bayes_Parameters.py script completed in $(format_time $SECONDS)"

python SelectFromModel_MultinomialNB.py \
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
 
echo "SelectFromModel_MultinomialNB.py script completed in $(format_time $SECONDS)"

python SelectFromModel_RandomForest.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir

echo "SelectFromModel_RandomForest.py script completed in $(format_time $SECONDS)"

python SelectFromModel_SDG.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
echo "SelectFromModel_SDG.py script completed in $(format_time $SECONDS)"
 
python Evaluate_Method_Accuracy.py\
  -r project_dir \
  -e expected_results_dir \
  -m mock_results_fp  \
  -s results_dirs\
  -o mock_dir\
  -u outdir
 
 echo "Evaluate_Method_Accuracy.py script completed in $(format_time $SECONDS)"
 echo "I'm Done"
 

