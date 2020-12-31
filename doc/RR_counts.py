import numpy as np
import time
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

#randoms
randoms = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/randoms1_south_nominal.npy')

def select_patch(cat, limits):
    
    patch = np.ones(len(cat[0]), bool)
    patch &= np.logical_and(cat[0] > limits[0], cat[0] < limits[1])
    patch &= np.logical_and(cat[1] > limits[2], cat[1] < limits[3])
    
    return patch

#

test = False
nbins=40
nthreads=32
bins = np.logspace(np.log10(0.001), np.log10(10.0), nbins + 1) #log bins

if test:
    limits = [160, 170, 5, 7]
    mask_ran = select_patch([randoms['RA'], randoms['DEC']], limits=limits)
    RAr, DECr = randoms['RA'][mask_ran], randoms['DEC'][mask_ran]
else:
    RAr, DECr = randoms['RA'], randoms['DEC']


start = time.time()
RR = DDtheta_mocks(0, nthreads, bins, RAr, DECr, RA2=RAr, DEC2=DECr)
end = time.time()
print('RR run time: %f sec' %(end - start))

np.save('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/pair_counts/RR_counts_randoms1_south_nominal', RR)