#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import os, sys
sys.path.insert(0, '/global/homes/q/qmxp55/DESI/bgstargets/py')
from io_ import get_random 
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
ap = ArgumentParser(description='Get SWEEPS sources in a single file')
ap.add_argument('-dr',
                help="Choose a data release from dr7, dr8, dr8-south, dr8-north, dr9d, dr9sv")
ap.add_argument('-Nran', default=3,
                help="Number of random files")
ap.add_argument('-OF',
                help="Filepath of output randoms catalogue")

ns = ap.parse_args()
if ns.dr == 'dr9sv': Nran = 1
else: Nran = ns.Nran
print('NRAN', Nran)

randoms = get_random(N=int(Nran), sweepsize=None, dr=ns.dr, dirpath=ns.OF)
