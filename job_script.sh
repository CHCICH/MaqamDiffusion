#!/bin/bash

#SBATCH --job-name=training the autoencoder
#SBATCH --account=cbe05

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-00:30:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbe05@mail.aub.edu

echo "Program will start executing in a bit"
module purge 
module load python/ai-4

source activate
python src/data/autoencoder_training.py 




