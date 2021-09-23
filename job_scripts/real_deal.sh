#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --job-name=classif_bnc_rb_bert_search_best_eval_correction
#SBATCH --time=5-00:00:00 ## Max time your script runs for (max is 5-00:00:00 | 5 days)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/classif_bnc_rb_bert_search_best_eval_correction.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

# Loading all necessary modules.
echo "Loading modules..."
module purge
module load 2020
#module load eb
#module load Python/3.7.5-foss-2019b
#module load Miniconda3
module load Anaconda3/2020.02

# Activate the conda environment
echo "Activating conda environment..."
source /home/lennertj/miniconda3/etc/profile.d/conda.sh
source activate base
conda info --envs
source activate /home/lennertj/miniconda3/envs/thesis_lisa

# Change directories
echo "Changing directory"
#cd $HOME/code/PPLM
cd $HOME/code/msc-ai-thesis

# Run your code
echo "Running python code..."
# declare an array variable
#declare -a arr=("gpt2-medium")
#declare -a arr=("bert-base-uncased")

# data fps
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/bnc/bnc_rb_full_generic_pplm.txt' \
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_generic_pplm.txt' \

#now loop through the above array
#for i in "${arr[@]}"
#do
#  for j in 1 2 3 4 5
#  do
#
#    python run_pplm_discrim_train.py --dataset 'generic' \
#          --dataset_fp '../msc-ai-thesis/data/bnc/bnc_rb_incl_stopwords_nonalph_generic_pplm.txt' \
#          --epochs 10 \
#          --batch_size 16 \
#          --log_interval 20000 \
#          --pretrained_model "$i"
#  done
#done

for seed in 1 2 3 4 5 6 7
do
  echo 'Starting new seed:'
  echo "$seed"

  python train_classifiers.py \
         --data 'bnc_rb' \
         --model_type 'bert' \
         --mode 'train' \
         --seed "$seed" \
         --batch_size 8 \
         --embedding_dim 512 \
         --hidden_dim 1024 \
         --num_layers 2 \
         --batch_first \
         --epochs 10 \
         --lr 0.0001 \
         --early_stopping_patience 4 \
         --train_frac 0.75 \
         --val_frac 0.15 \
         --test_frac 0.1 \
         --log_interval 1000 \
         --no_tb \
         --w_sampling \
         --w_loss \
         --dropout 0.1
done

#python n_gram_classifiers.py \
#       --dataset 'blog' \
#       --n_grams 1 2 3 \
#       --seeds 1 2 3 4 5 \
#       --subset_size -1 \
#       --test_size 0.2

#python case_analysis.py