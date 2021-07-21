#!/bin/bash

#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --job-name=debugger
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/debugger.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

# Loading all necessary modules.
echo "Loading modules..."
module purge
module load eb
module load 2019
module load Python/3.7.5-foss-2019b
module load Miniconda3

# Activate the conda environment
echo "Activating conda environment..."
source activate base
conda activate thesis_lisa2

# Change directories
echo "Changing directory"
cd $HOME/code/PPLM

# Run your code
echo "Running python code..."
python run_pplm_discrim_train.py --dataset 'generic' \
      --dataset_fp '/home/lennertj/code/msc-ai-thesis/data/bnc/bnc_rb_small_generic_pplm.txt' \
      --epochs 1 \
      --batch_size 64 \
      --pretrained_model 'bert-base-uncased'