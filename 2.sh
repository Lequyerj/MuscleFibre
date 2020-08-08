#!/bin/bash
# My first script

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate muscles
python 2.py
