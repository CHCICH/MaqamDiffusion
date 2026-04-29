#!/bin/bash

#SBATCH --job-name=trainingautoencoder
#SBATCH --account=cbe05

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-01:40:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbe05@mail.aub.edu

echo "Program will start executing in a bit"
module purge 
module load python/ai-4


source .diffusion/bin/activate
python src/testing/autoencoder_training.py 




