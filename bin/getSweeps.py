#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import os, sys
sys.path.insert(0, '/global/homes/q/qmxp55/DESI/bgstargets/py')
from io_ import get_sweep_whole 
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
ap = ArgumentParser(description='Get SWEEPS sources in a single file')
ap.add_argument('-dr',
                help="Choose a data release from dr7, dr8, dr8-south, dr8-north, dr9d")
ap.add_argument('-rlim', default=21.0,
                help="Choose r-mag fainter limit")
ap.add_argument('-patch', default=None,
                help="Script can extract Sweep targets from a rectangular patch of the form [RAmin, RAmax, DECmin, DECmax]")
ap.add_argument('-OF',
                help="Filepath of sweep catalogue")

ns = ap.parse_args()

if ns.patch is not None:
    patch_ = ns.patch.split(",")
    patch = [np.float(patch_[0]), np.float(patch_[1]), np.float(patch_[2]), np.float(patch_[3])]
else:
    patch = None
    
if ns.rlim == 'None': rlim = None
else: rlim = np.float(ns.rlim)


#print('===========')
print('patch',patch)
#print('dr', ns.dr)
#print('dr type', type(ns.dr))
print('rlim', ns.rlim)
#print('filepath', ns.f)
#print('===========')

cat =  get_sweep_whole(patch=patch, dr=ns.dr, rlimit=rlim, maskbitsource=False, bgsbits=False, opt='1', sweepdir=ns.OF)