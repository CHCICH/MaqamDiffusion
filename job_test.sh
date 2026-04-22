#!/bin/bash


#SBATCH --job-name=testjob
#SBATCH --account=cbe05

#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=0-00:30:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cbe05@mail.aub.edu


echo "Hello World" 


