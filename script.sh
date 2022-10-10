#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30Gb
#SBATCH --output=metu_trial.out
#SBATCH --time=60:00:00
 
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/SE/BMIG-6202-MSR/qiime2-2022.2 #change to the environment where qiime2 is downloaded.

wd="/scratch/metu/input/" #temporary working directory. Change to your own path.
if [ ! -d "$wd" ]; then
  mkdir -p "$wd"
else
  rm -rf "$wd"/*
fi
echo "working directory created/emptied"

cp ~/Feature-Selection-Qiime2/Naive_Bayes_Parameters.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_MultinomialNB.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_RandomForest.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_SDG.py "$wd"
cp ~/Feature-Selection-Qiime2/Evaluate_Method_Accuracy.py "$wd"
cp -r ~/Feature-Selection-Qiime2/tax_credit "$wd"
cp -r ~/tax-credit-data "$wd"

cd "$wd"

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

python Naive_Bayes_Parameters.py \
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
echo "Naive_Bayes_Parameters.py script completed after $(format_time $SECONDS)"

python SelectFromModel_MultinomialNB.py \
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
 
echo "SelectFromModel_MultinomialNB.py script completed after $(format_time $SECONDS)"

python SelectFromModel_RandomForest.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir

echo "SelectFromModel_RandomForest.py script completed after $(format_time $SECONDS)"

python SelectFromModel_SDG.py\
  -r project_dir \
  -a analysis_name \
  -d data_dir \
  -e precomputed_dir\
  -s results_dir\
  -f reference_database_dir
  
echo "SelectFromModel_SDG.py script completed after $(format_time $SECONDS)"
 
python Evaluate_Method_Accuracy.py\
  -r project_dir \
  -e expected_results_dir \
  -m mock_results_fp  \
  -s results_dirs\
  -o mock_dir\
  -u outdir
 
 echo "Evaluate_Method_Accuracy.py script completed after $(format_time $SECONDS)"
 echo "That means this whole script took $(format_time $SECONDS) to run"
 echo "I'm Done"
 

