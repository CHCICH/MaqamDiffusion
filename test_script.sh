#!/bin/bash

#SBATCH --job-name=trainingautoencoder
#SBATCH --account=cbe05

# Fix 1: Change partition from normal to gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000

# Fix 2: Correct the GPU resource name to the v100d16q queue
#SBATCH --gres=gpu:v100d16q:1
#SBATCH --time=0-00:20:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbe05@mail.aub.edu

echo "Program will start executing in a bit"
module purge 
module load python/ai-4

source .diffusion/bin/activate
python results/jmw.py
