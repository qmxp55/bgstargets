#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import os, sys
sys.path.insert(0, '/global/homes/q/qmxp55/DESI/omarlibs/bgstargets/py')
from io_ import get_sweep_whole 
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
ap = ArgumentParser(description='Generates DESI BGS target bits from Legacy Surveys sweeps')
ap.add_argument('-dr',
                help="Choose a data release from dr7, dr8, dr8-south, dr8-north, dr9d")
ap.add_argument('-rlim', default=21,
                help="Choose r-mag fainter limit")
ap.add_argument('-patch', default=None,
                help="Script can extract Sweep targets from a rectangular patch of the form [RAmin, RAmax, DECmin, DECmax]")

ns = ap.parse_args()

if ns.patch is not None:
    patch_ = ns.patch.split(",")
    patch = [np.float(patch_[0]), np.float(patch_[1]), np.float(patch_[2]), np.float(patch_[3])]
else:
    patch = None


print('===========')
print(patch)
print(ns.dr)
print(type(ns.dr))
print('===========')

cat =  get_sweep_whole(patch=patch, dr='dr8-south', rlimit=np.float(ns.rlim), maskbitsource=False, bgsbits=True, opt='1', version='2.0')