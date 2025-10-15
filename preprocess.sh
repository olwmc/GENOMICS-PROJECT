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

python dataset.py \
  --hg19_fa       data/hg19/hg19.fa \
  --chrom_sizes   data/hg19/hg19.chrom.sizes \
  --blacklist_bed data/blacklist/hg19-blacklist.v2.bed \
  --epiphany_h5   data/epiphany/GM12878_X.h5 \
  --epiphany_y    data/epiphany/GM12878_y.pickle \
  --out_root      data/siamese_5kb_out
