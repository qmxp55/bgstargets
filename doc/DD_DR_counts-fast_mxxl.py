#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import time
import Corrfunc
import random
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

import warnings
warnings.filterwarnings('ignore')

#from argparse import ArgumentParser
#ap = ArgumentParser(description='Get count pairs')
#ap.add_argument('-bin', help="choose colour bin in targets")

#ns = ap.parse_args()
#print('===== %s =====' %(ns.bin))

#targets
targets = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/MXXL/galaxy_catalogue_south.npy')
#randoms
randoms = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/randoms.npy')
#split randoms

#
colours = {}
g_r = targets['g_r_obs_smooth'.upper()]
val = np.array([0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
for num, i in enumerate(val[:-1]):
    colours['%s_%s' %(str(val[num]), str(val[num+1]))] = (g_r > val[num]) & (g_r < val[num+1]) & (np.isfinite(g_r))

#get the index array
idx = [i for i in range(len(randoms['RA']))]
#shuffle the array
random.shuffle(idx)

#split array in chunks
chunks = {}
n = 16
for i in range(n):
    chunks['%i' %(i)] = idx[i::n]
    
#
nbins=40
nthreads=16
bins = np.logspace(np.log10(0.001), np.log10(10.0), nbins + 1) #log bins

#cbin = '0.0_0.5'

for key in colours.keys():
    
    pairs2 = {}
    
    val = colours[key]
    RA1, DEC1 = targets['RA'][val].astype('float64'), targets['DEC'][val].astype('float64')
    
    start = time.time()
    pairs2['DD_%s' %(key)] = DDtheta_mocks(1, nthreads, bins, RA1, DEC1)
    end = time.time()
    print('DD run time: %f sec' %(end - start))
    
    for keyR, val in chunks.items():
        
        RAr, DECr = randoms['RA'][val].astype('float64'), randoms['DEC'][val].astype('float64')
        start = time.time()
        pairs2['DR_%s_%s' %(key, keyR)] = DDtheta_mocks(0, nthreads, bins, RAr, DECr, RA2=RA1, DEC2=DEC1)
        end = time.time()
        print('DR_%s_%s run time: %f sec' %(key, keyR, end - start))
        
    np.save('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/pair_counts/DD_DR_decals_16_%s_MXXL.npy' %(key), pairs2)