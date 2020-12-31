#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import os, sys
sys.path.insert(0, '/global/homes/q/qmxp55/DESI/bgstargets/py')
from io_ import bgsmask, get_reg, get_svfields, get_svfields_fg, get_svfields_ij

import healpy as hp
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.table import Table
import time
import fitsio

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
ap = ArgumentParser(description='Generates DESI BGS target bits from Legacy Survey sweeps')

ap.add_argument('-IF',
                help="Input filepath to add EXTRA info")
ap.add_argument('-OF',
                help="Outout Filepath of extra catalogue including healpix info, galactic coordinates and regions flags (DES, DECALS, DESI, NORTH, SOUTH, SVFIELDS)")
ap.add_argument('-random', default=False,
                help="True if using random catalogue")

ns = ap.parse_args()

start = time.time()
    
if ns.IF[-4:] == 'fits':
    df = fitsio.read(ns.IF)
elif ns.IF[-3:] == 'npy':
    df = np.load(ns.IF)
else:
    raise ValueError('%s is not a supported file. Usea file in either .fits or .npy format.' %(ns.IF))

cwd = os.path.dirname(os.path.realpath(ns.OF))

#save BGSBITS key to text file
if not ns.random:
    f = open(cwd+'/'+'bgskey.txt',"w")
    f.write( str(bgsmask()))
    f.close()

extra = Table()

extra['RA'] = df['RA']
extra['DEC'] = df['DEC']

#angle to healpy pixels array
nside = 256
extra['hppix'] = hp.ang2pix(nside,(90.-df['DEC'])*np.pi/180.,df['RA']*np.pi/180.,nest=True)
print('healpix DONE...')
c = SkyCoord(df['RA']*units.degree,df['DEC']*units.degree, frame='icrs')
extra['b'] = c.galactic.b.value # galb coordinate
extra['l'] = c.galactic.l.value # galb coordinate
print('galactic coordinates DONE...')

regs = ['des', 'decals', 'north', 'desi', 'south']
for i in regs:
    reg_ = get_reg(reg=i, hppix=extra['hppix'])
    extra[i] = reg_
    print(i, 'DONE...')
    
#extra['svfields_fg'] = get_svfields_fg(df['RA'],df['DEC'])
#print('svfields_fg', 'DONE...')

#extra['svfields_ij'] = get_svfields_ij(df['RA'],df['DEC'], survey='all')
#print('svfields_ij', 'DONE...')

#extra['svfields'] = get_svfields(df['RA'],df['DEC'])
#print('svfields', 'DONE...')

# save astropy table as npy file
np.save(ns.OF, extra)

end = time.time()
print('Total run time: %f sec' %(end - start))
