import numpy as np
import time
import Corrfunc
import random
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

#randoms
randoms = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/randoms.npy')

#split randoms

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
ran_pairs = {}
nbins=40
nthreads=16
bins = np.logspace(np.log10(0.001), np.log10(10.0), nbins + 1) #log bins

for keyR, val in chunks.items():

    RAr, DECr = randoms['RA'][val], randoms['DEC'][val]

    start = time.time()
    ran_pairs['RR_%s' %(keyR)] = DDtheta_mocks(0, nthreads, bins, RAr, DECr, RA2=RAr, DEC2=DECr)
    end = time.time()
    print('RR_%s run time: %f sec' %(keyR, end - start))
    
    
np.save('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/pair_counts/RR_south_16.npy', ran_pairs)
