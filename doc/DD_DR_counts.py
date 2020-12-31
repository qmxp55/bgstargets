import numpy as np
import time
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

#targets
targets = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/south_nominal.npy')
#randoms
randoms = np.load('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/randoms1_south_nominal.npy')

#
colours = {}
g_r = targets['GMAG']-targets['RMAG']
val = np.linspace(0,2,5)
for num, i in enumerate(val[:-1]):
    colours['%s_%s' %(str(val[num]), str(val[num+1]))] = (g_r > val[num]) & (g_r < val[num+1]) & (np.isfinite(g_r))
    
for key, val in colours.items():
    print('%s: \t %i' %(key, np.sum(val)))

def select_patch(cat, limits):
    
    patch = np.ones(len(cat[0]), bool)
    patch &= np.logical_and(cat[0] > limits[0], cat[0] < limits[1])
    patch &= np.logical_and(cat[1] > limits[2], cat[1] < limits[3])
    
    return patch

#

test = False
#
limits = [160, 180, 5, 7]
pairs = {}
nbins=40
nthreads=32
bins = np.logspace(np.log10(0.001), np.log10(10.0), nbins + 1) #log bins

if test:
    mask_tar = select_patch([targets['RA'], targets['DEC']], limits=limits)
    mask_ran = select_patch([randoms['RA'], randoms['DEC']], limits=limits)
else:
    mask_tar = np.ones(len(targets['RA']), dtype=bool)
    mask_ran = np.ones(len(randoms['RA']), dtype=bool)


for key, val in colours.items():
    
    keep_tar = (mask_tar) & (val)
    RA1, DEC1 = targets['RA'][keep_tar], targets['DEC'][keep_tar]
    RAr, DECr = randoms['RA'][mask_ran], randoms['DEC'][mask_ran]

    start = time.time()
    pairs['DD_%s' %(key)] = DDtheta_mocks(1, nthreads, bins, RA1, DEC1)
    end = time.time()
    print('DD run time: %f sec' %(end - start))

    start = time.time()
    pairs['DR_%s' %(key)] = DDtheta_mocks(0, nthreads, bins, RAr, DECr, RA2=RA1, DEC2=DEC1)
    end = time.time()
    print('DR run time: %f sec' %(end - start))

np.save('/global/cscratch1/sd/qmxp55/bgstargets_output/dr9/clustering/pair_counts/DD_DR_counts_randoms1_south_nominal', pairs)