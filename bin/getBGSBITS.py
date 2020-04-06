#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import os, sys
sys.path.insert(0, '/global/homes/q/qmxp55/DESI/bgstargets/py')
from io_ import getBGSbits, bgsmask, get_reg

import healpy as hp
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.table import Table

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
ap = ArgumentParser(description='Generates DESI BGS target bits from Legacy Survey sweeps')
#ap.add_argument('-dr',
#                help="Choose a data release from dr7, dr8, dr8-south, dr8-north, dr9d")

ap.add_argument('-mycat', default=True,
                help="True if input catalogue is the single sweep file created by `get_sweep_whole` function. False if using external catalogue.")
ap.add_argument('-getmycat', default=False,
                help="True if using external catalogue and want output catalogue like `get_sweep_whole`")
ap.add_argument('-IF',
                help="Input filepath to add BGSBITS info")
ap.add_argument('-OF',
                help="Outout Filepath of sweep catalogue")

ns = ap.parse_args()

#get SWEEPS with BGSBITS column
df = getBGSbits(mycatpath=ns.IF, outdir=ns.OF, mycat=ns.mycat, getmycat=ns.getmycat)
