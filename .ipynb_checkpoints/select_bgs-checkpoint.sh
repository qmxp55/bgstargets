#!/bin/bash

#module load python/3.7-anaconda-2019.07
#module unload desimodules
#source /project/projectdirs/desi/software/desi_environment.sh 18.7

module load python/3.7-anaconda-2019.07
source deactivate
conda deactivate 
conda activate DESI_BGS_omar

#python bin/select_bgs.py -dr 'dr8' -rlim 21 -patch 174.,186.,-3.,2.
python bin/select_bgs.py -dr 'dr8-south' -rlim 21.0