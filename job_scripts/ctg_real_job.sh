#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --job-name=ctg_bow_100mcw_WS_baseline_50_50_redo_old
#SBATCH --time=5-00:00:00 ## Max time your script runs for (max is 5-00:00:00 | 5 days)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/ctg_bow_100mcw_WS_baseline_50_50_redo_old.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

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
source activate /home/lennertj/miniconda3/envs/thesis_lisa2

# Change directories
echo "Changing directory"
#cd $HOME/code/PPLM
cd $HOME/code/msc-ai-thesis

# Run your code
echo "Running python code..."

# for BoW-based
for seed in 2021
do
  for length in 6 12 24 30 36 42 48 54 60
  do
    python plug_play/run_pplm.py \
           --pretrained_model 'gpt2-medium' \
           --num_samples 15 \
           --bag_of_words 'plug_play/wordlists/bnc_rb_ws_100_most_common.txt' \
           --length $length \
           --seed $seed \
           --sample \
           --class_label 1 \
           --verbosity "quiet" \
           --uncond
  done
done

# for discrim-based
#for seed in 2021
#do
#  for length in 8 16 32 64
#  do
#    python plug_play/run_pplm.py \
#           --pretrained_model 'gpt2-medium' \
#           --cond_text 'My first impression' \
#           --uncond \
#           --num_samples 30 \
#           --discrim 'generic' \
#           --length $length \
#           --seed $seed \
#           --sample \
#           --class_label 1 \
#           --verbosity "quiet" \
#           --discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" \
#           --discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json"
#  done
#done

# declare an array variable
#declare -a arr=("gpt2-medium")
#declare -a arr=("bert-base-uncased")

# data fps
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/bnc/bnc_rb_full_generic_pplm.txt' \
# --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_generic_pplm.txt' \

## now loop through the above array
#for i in "${arr[@]}"
#do
#  for j in 1 2 3 4 5
#  do
#
#    python run_pplm_discrim_train.py --dataset 'generic' \
#          --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/blogs_kaggle/blog_generic_pplm.txt' \
#          --epochs 5 \
#          --batch_size 16 \
#          --log_interval 20000 \
#          --pretrained_model "$i"
#  done
#done

#for seed in 6 8 9 10 11 12 13
#do
#  echo 'Starting new seed:'
#  echo "$seed"
#
#  python train_classifiers.py \
#         --data 'bnc_rb' \
#         --model_type 'bert' \
#         --mode 'train' \
#         --seed "$seed" \
#         --batch_size 4 \
#         --embedding_dim 128 \
#         --hidden_dim 256 \
#         --num_layers 2 \
#         --batch_first \
#         --epochs 10 \
#         --lr 0.001 \
#         --early_stopping_patience 3 \
#         --train_frac 0.75 \
#         --val_frac 0.15 \
#         --test_frac 0.1 \
#         --log_interval 10000 \
#         --no_tb
#done