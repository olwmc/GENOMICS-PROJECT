#!/bin/bash

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --mem=16G

#SBATCH --time=24:00:00

#SBATCH -J job

#SBATCH -o out/%j.out

#SBATCH -e errors/%j.err

module load python/3.11

source ~/genomics_env/bin/activate

python dataset.py 
  
