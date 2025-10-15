import argparse, os, sys, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from pyfaidx import Fasta
import torch
from torch.utils.data import Dataset

# Sanity check that all three data sources have roughly the same size

def _open_epiphany_h5(h5_path):
    f = h5py.File(h5_path, "r")
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            return f, f[k]


def main():
    f, dset = _open_epiphany_h5('data/epiphany/GM12878_X.h5')
    shape = dset.shape
    print("epiphany shape: ", shape[1]*100)
    print("hic: ", 49850*5000)
    fa = Fasta("data/hg38/hg38.fa", as_raw=True)  # will build hg38.fa.fai index automatically
    for chrom in fa.keys():
        print(chrom, len(fa[chrom]))
        break 

if __name__ == "__main__":
    main()