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

# # precompute dataset
# python dataset.py \
#     --mode pair --hic-norm KR --oe-metric oe \
#     --bin-edges 25000,100000,400000,1000000,10000000 \
#     --pos-quantile 0.7 --neg-quantile 0.3 --num-negatives 8 \
#     --min-distance-bp 25000 --pairs-per-batch 64 \
#     --patch-size-bp 100 \
#     --cache-save-dir /users/jrober48/data/jroberts/2952-G/mango_precomputed \

# Use precomputed dataset
python dataset.py \
    --mode pair --hic-norm KR --oe-metric oe \
    --bin-edges 25000,100000,400000,1000000,10000000 \
    --pos-quantile 0.7 --neg-quantile 0.3 --num-negatives 8 \
    --min-distance-bp 25000 --pairs-per-batch 64 \
    --patch-size-bp 100 \
    --cache-load-dir /users/jrober48/data/jroberts/2952-G/mango_precomputed


