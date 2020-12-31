#!/bin/bash

module load python/3.7-anaconda-2019.07
source deactivate
conda deactivate 
conda activate DESI_BGS_omar

python doc/RR_counts-fast.py