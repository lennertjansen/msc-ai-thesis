#!/bin/bash

##SBATCH --partition=gpu
#SBATCH -p gpu_titanrtx_shared ## Select the partition. This one is almost always free, and has TitanRTXes (much RAM)
#SBATCH --nodes=1
##SBATCH --gpus-per-node=1
#SBATCH --job-name=scripted_dialogue_generation_food_turnwise_old
#SBATCH --time=5-00:00:00 ## Max time your script runs for (max is 5-00:00:00 | 5 days)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lennertjansen95@gmail.com
#SBATCH -o /home/lennertj/code/msc-ai-thesis/SLURM/output/scripted_dialogue_generation_food_turnwise_old.%A.out ## this is where the terminal output is printed to. %j is root job number, %a array number. try %j_%a ipv %A (job id)

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

## for BoW-based
#declare -a arr=("Tell me about your holidays. Sure! I went to Greece and had a very fun time." "Tell me about your favorite food. Of course. I love pasta!." "Have you seen the news lately? I can't believe what I saw.")
declare -a arr=("Hello, how are you?<|endoftext|>")
#declare -a young_prompts=("Tell me about your favourite food. I love sushi and Japanese food in general. What about you?" "Anything salmon. I am not so much of a spicy food lover, however. I love food too." "Me too! Meat and fish all the way." "I agree! A lot of great fish and meat. And it is also nice and lean." "Tell me about your favourite food. Too many to choose. I think one of my favorites for sure is a traditional stuffed pigeon from home. We eat pigeon a lot over there and it's usually stuffed with the most amazing herb rice! I usually eat it for special ocassions though, as it is quite hard to prepare and takes quite some time." "Hmm.. Probably a traditional egyptian one too. It's called konafa." "Tell me about yours?" "That's always a good idea! Me too. THe more local/fresh, the better. Tell me about your best dining experience!" "Tell me about your favourite food. My favorite food is fried chicken with French fries." "We share something in common." "I would describe myself as a fun-loving person. What about you?" "What is your favorite drink?" "I love apple juice.")
declare -a old_prompts=("Tell me about your favourite food. Well, I am very fun with Mediterranean food, and especially food with herbs. Lot of fruits, nuts, especially in the summer weather." "I love seafood and fish, except I do not like chips! I much prefer olive fish accompanied with green salad." "I actually I am not fond of cheese and it is not a food that I include in my diet" "It's not different at all. My favourite food is what I usually consume." "Tell me about your favourite food. Healthy or non healthy food, which would you prefer me to talk about?" "Well I love fish, all kinds of fish, I try and eat meat sparingly, love fruit and vegetables, so I ought to be very healthy, but unfortunately I have a sweet tooth for Chocolate." "Well I love Cadburys, especially Fruit and Nut Chocolate, thinking the fruit and nut is somehow healthy for me. I also like Dark Chocolate as I know that should be better than milk chocolate, especially for the heart." "My allergies, well I thought that was typed as arteries, my arteries are probably clogged with chocolate, so not as healthy as I ought to be." "Salmon, which is fish, prawn, which is fish also, in terms of meat/animal, lamb, chicken and beef/steak. But as I mentioned earlier I try and eat more fruit and vegetables." "Tell me about your favourite food. I enjoy spicy foods. I have a curry quite a lot." "I also enjoy fish and chips sometimes , but not too often as it is fattening." "You need to have vegtables with your chips as it adds to the taste." "Do you like halloumi? Its one of my favorites.")

# for discrim-based
for prompt in "${old_prompts[@]}"
do
  echo "NEW PROMPT ALERT WOOOWOOOOWOOOOOOWOOOOOOO"
  for seed in 2021
  do
    for length in 35
    do
      python plug_play/run_pplm.py \
             --pretrained_model 'gpt2-medium' \
             --num_samples 3 \
             --bag_of_words "plug_play/wordlists/bnc_old_mcwu_ws_pct_85.txt" \
             --length $length \
             --seed $seed \
             --sample \
             --class_label 1 \
             --verbosity "quiet" \
             --cond_text "$prompt"
    done
  done
done

#declare -a arr=("Tell me about your holidays. Sure! I went to Greece and had a very fun time." "Tell me about your favorite food. Of course. I love pasta!." "Have you seen the news lately? I can't believe what I saw.")
#declare -a arr=("Hello, how are you?<|endoftext|>")
#
#
## for discrim-based
#for prompt in "${arr[@]}"
#do
#  for seed in 2021
#    do
#
#    for length in 6 12 24 30 36 42 48 54 60
#    do
#
#      python plug_play/run_pplm.py \
#             --pretrained_model 'microsoft/DialoGPT-medium' \
#             --cond_text "$prompt" \
#             --num_samples 30 \
#             --discrim 'generic' \
#             --length $length \
#             --seed $seed \
#             --sample \
#             --class_label 0 \
#             --verbosity "quiet" \
#             --discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" \
#             --discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json" \
#             --num_iterations 0 \
#             --stepsize 0
#    done
#  done
#done



# Weights and metadata for discriminators

# gpt2-medium
#--discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" \
#--discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json"

# microsoft/dialogpt-medium
#--discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" \
#--discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json"

# for length in 6 12 24 30 36 42 48 54 60 # Lengths used for main results table until now (Monday, 27 September 2021)

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

#python plug_play/run_pplm.py --pretrained_model 'microsoft/DialoGPT-medium' --cond_text 'Hello, how are you? Fine, thanks. And you?' --num_samples 5 --discrim 'generic' --length 15 --seed 2021 --sample --class_label 0 --verbosity "verbose" --discrim_weights "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_epoch_17.pt" --discrim_meta "plug_play/discriminators/dialogpt-medium/generic_pm_microsoft-DialoGPT-medium_ml_512_lr_0.0001_classifier_head_meta.json" --num_iterations 0 --stepsize 0

# plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json
# plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt

# plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt

# python plug_play/run_pplm.py --pretrained_model 'gpt2-medium' --cond_text 'A: Hello, how are you? B: Fine, thanks. And you? A:' --num_samples 5 --discrim 'generic' --length 15 --seed 2021 --sample --class_label 1 --verbosity "verbose" --discrim_weights "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_epoch_19.pt" --discrim_meta "plug_play/discriminators/gpt2_incl_sw_nac/generic_pm_gpt2-medium_ml_512_lr_0.0001_classifier_head_meta.json" --num_iterations 0 --stepsize 0
# python plug_play/run_pplm.py --pretrained_model 'gpt2-medium' --bag_of_words "plug_play/wordlists/bnc_young_mcwu_ws_pct_85.txt" --cond_text 'A: Hello, how are you? B: Fine, thanks. And you? A:' --num_samples 5 --length 15 --seed 2021 --sample --class_label 0 --verbosity "verbose"