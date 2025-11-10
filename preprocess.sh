#!/bin/bash

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --mem=64G

#SBATCH --time=24:00:00

#SBATCH -J job

#SBATCH -o out/%j.out

#SBATCH -e errors/%j.err

module load python/3.11

source ~/genomics_env/bin/activate

python dataset.py \
    --mode pair --hic-norm KR --oe-metric oe \
    --chroms chr1 \
    --bin-edges 25000,100000,400000,1000000,10000000 \
    --pos-quantile 0.7 --neg-quantile 0.3 --num-negatives 8 \
    --min-distance-bp 25000 --pairs-per-batch 32 \
    --patch-size-bp 100 --token-mode thin --emit-pos-ids
  
