#
import numpy as np
import sys, os, time, argparse, glob
import fitsio
from astropy.table import Table
import astropy.io.fits as fits
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy import units as u
from veto import veto, veto_ellip, match

from cuts import getGeoCuts, getPhotCuts, get_bgs, get_bgs_sv

#from QA import circular_mask_radii_func

def getBGSbits(mycatpath=None, outdir=None, mycat=True, getmycat=False, tractor=False):
    
    import time
    start = time.time()
    
    if mycatpath[-4:] == 'fits':
        df = fitsio.read(mycatpath)
    else:
        df = np.load(mycatpath)
        
        
    tab = Table()
    
    if getmycat:
        for col in df.dtype.names:
            #if tractor: col = col.upper()
            if (col[:4] == 'FLUX') & (col[:9] != 'FLUX_IVAR') & (col[:6] != 'FLUX_W'): tab[col[-1:]+'MAG'] = flux_to_mag(df['FLUX_'+col[-1:]]/df['MW_TRANSMISSION_'+col[-1:]])
            elif col[:2] == 'MW': continue
            elif col == 'FIBERFLUX_R': tab['RFIBERMAG'] = flux_to_mag(df[col]/df['MW_TRANSMISSION_R'])
            elif col == 'GAIA_PHOT_G_MEAN_MAG': tab['G'] = df[col]
            elif col == 'GAIA_ASTROMETRIC_EXCESS_NOISE': tab['AEN'] = df[col]
            else: tab[col] = df[col]
        tab['FLUX_R'] = df['FLUX_R']
    else:
        for col in df.dtype.names:
            #if not tractor: col = col.upper()
            #col = col.upper()
            if col == 'BGSBITS': continue
            tab[col.upper()] = df[col]
            
    # create BGSBITS: bits associated to selection criteria
  
    geocuts = getGeoCuts(tab)
    photcuts = getPhotCuts(tab, mycat=mycat)
    bgscuts = geocuts
    bgscuts.update(photcuts)
            
    BGSBITS = np.zeros_like(tab['RA'], dtype='i8')
    BGSMASK = {}
            
    print('---- BGSMASK key: ---- ')
    for bit, key in enumerate(bgscuts.keys()):
                
        BGSBITS |= bgscuts[key] * 2**(bit)
        BGSMASK[key] = bit
        print('\t %s, %i, %i' %(key, bit, 2**bit))

    bgs_any, bgs_bright, bgs_faint = get_bgs(tab, mycat=mycat)
    bgs_sv_any, bgs_sv_bright, bgs_sv_faint, bgs_sv_faint_ext, bgs_sv_fibmag, bgs_sv_lowq = get_bgs_sv(tab, mycat=mycat)
            
    BGSsel = {'bgs_any':bgs_any, 'bgs_bright':bgs_bright, 'bgs_faint':bgs_faint}
    BGSSVsel = {'bgs_sv_any':bgs_sv_any, 
                'bgs_sv_bright':bgs_sv_bright, 
                'bgs_sv_faint':bgs_sv_faint,
                'bgs_sv_faint_ext':bgs_sv_faint_ext,
                'bgs_sv_fibmag':bgs_sv_fibmag,
                'bgs_sv_lowq':bgs_sv_lowq}
            
    for num, key in enumerate(BGSsel.keys()):
        BGSBITS |= BGSsel[key] * 2**(20+num)
        BGSMASK[key] = 20+num
        print('\t %s, %i, %i' %(key, 20+num, 2**(20+num)))
                
    for num, key in enumerate(BGSSVsel.keys()):
        BGSBITS |= BGSSVsel[key] * 2**(30+num)
        BGSMASK[key] = 30+num
        print('\t %s, %i, %i' %(key, 30+num, 2**(30+num)))
            
    tab['BGSBITS'] = BGSBITS
            
    # sanity check...
    print('---- Sanity Check ---- ')
    for bit, key in zip(BGSMASK.values(), BGSMASK.keys()):
        if key in list(BGSsel.keys()): print('\t %s, %i, %i' %(key, np.sum(BGSsel[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
        elif key in list(BGSSVsel.keys()): print('\t %s, %i, %i' %(key, np.sum(BGSSVsel[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
        else: print('\t %s, %i, %i' %(key, np.sum(bgscuts[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
           
    if outdir is not None:
        np.save(outdir, tab)

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    
    return tab
    

def get_sweep_whole(patch=None, dr='dr8-south', rlimit=None, maskbitsource=False, bgsbits=False, opt='1', sweepdir='/global/cscratch1/sd/qmxp55/bgstargets_output/'):
    """
    Extract data from DECaLS DR7 SWEEPS files only.
    
    Parameters
    ----------
    patch: :class:`array-like`
        Sky coordinates in RA and DEC of the rectangle/square patch in format [RAmin, RAmax, DECmin, DECmax]
    rlimit: :class:`float`
        magnitude limit of data in the r-band with extinction correction
    
    Returns
    -------
    Subsample catalogue of SWEEP data.
    The subsample catalogue will be also stored with name 'sweep_RAmin_RAmax_DECmin_DECmax_rmag_rlimit' and numpy format '.npy'
    
    """
    import time
    start = time.time()
    
    namelab = []
    if patch is not None: namelab.append('_'.join([str(i) for i in patch]))
    if rlimit is not None: namelab.append('r%s' %(str(rlimit)))
    if maskbitsource: namelab.append('maskbitsource')
    #print(namelab)
    
    if (len(namelab) > 0) & (patch is None): sweep_file_name = '%s_sweep_whole_%s' %(dr, '_'.join(namelab))
    elif (len(namelab) > 0) & (patch is not None): sweep_file_name = '%s_sweep_%s' %(dr, '_'.join(namelab))
    else: sweep_file_name = '%s_sweep_whole' %(dr)
        
    sweep_file = os.path.isfile(sweepdir+sweep_file_name+'.npy')
    sweep_dir_dr7 = os.path.join('/global/project/projectdirs/cosmo/data/legacysurvey/','dr7', 'sweep', '7.1')
    sweep_dir_dr8south = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
    sweep_dir_dr8north = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
    sweep_dir_dr9dsouth = '/global/cscratch1/sd/desimpp/dr9d/south/sweep'
    sweep_dir_dr9dnorth = '/global/cscratch1/sd/desimpp/dr9d/north/sweep'
    sweep_dir_dr9svsouth = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9sv/south/sweep'
    sweep_dir_dr9svnorth = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9sv/north/sweep'
    sweep_dir_dr9fsouth = '/global/cscratch1/sd/landriau/dr9f/south/sweep'
    sweep_dir_dr9fnorth = '/global/cscratch1/sd/landriau/dr9f/north/sweep'
    sweep_dir_dr9gsouth = '/global/cscratch1/sd/landriau/dr9g/south/sweep'
    sweep_dir_dr9gnorth = '/global/cscratch1/sd/landriau/dr9g/north/sweep'
    sweep_dir_dr9isouth = '/global/cscratch1/sd/adamyers/dr9i/south/sweep'
    sweep_dir_dr9inorth = '/global/cscratch1/sd/adamyers/dr9i/north/sweep'
    sweep_dir_dr9jsouth = '/global/cscratch1/sd/adamyers/dr9j/south/sweep'
    sweep_dir_dr9jnorth = '/global/cscratch1/sd/adamyers/dr9j/north/sweep'
    
    if not sweep_file:
        if dr is 'dr7': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr7, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr8-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr8south, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr8-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr8north, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9sv-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9svsouth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9sv-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9svnorth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9f-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9fsouth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9f-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9fnorth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9g-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9gsouth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9g-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9gnorth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9i-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9isouth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9i-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9inorth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9j-south': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9jsouth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr9j-north': df = cut_sweeps(patch=patch, sweep_dir=sweep_dir_dr9jnorth, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
            
        #elif (dr is 'dr8') or (dr is 'dr9d'):
        #    if dr is 'dr8':
        #        sweep_north = sweep_dir_dr8north
        #        sweep_south = sweep_dir_dr8south
        #    elif dr is 'dr9d':
        #        sweep_north = sweep_dir_dr9dnorth
        #        sweep_south = sweep_dir_dr9dsouth
        #        
        #    print('getting data in the SOUTH')
        #    dfsouth = cut_sweeps(patch=patch, sweep_dir=sweep_south, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        #    print('getting data in the NORTH')
        #    dfnorth = cut_sweeps(patch=patch, sweep_dir=sweep_north, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        #    
        #    if dfnorth is None: df = dfsouth
        #    elif dfsouth is None: df = dfnorth
        #    else: df = np.concatenate((dfnorth, dfsouth))
                
        tab = Table()
        for col in df.dtype.names:
            if (col[:4] == 'FLUX') & (col[:9] != 'FLUX_IVAR'): tab[col[-1:]+'MAG'] = flux_to_mag(df['FLUX_'+col[-1:]]/df['MW_TRANSMISSION_'+col[-1:]])
            elif col[:2] == 'MW': continue
            elif col == 'FIBERFLUX_R': tab['RFIBERMAG'] = flux_to_mag(df[col]/df['MW_TRANSMISSION_R'])
            elif col == 'GAIA_PHOT_G_MEAN_MAG': tab['G'] = df[col]
            elif col == 'GAIA_ASTROMETRIC_EXCESS_NOISE': tab['AEN'] = df[col]
            else: tab[col] = df[col]
        tab['FLUX_R'] = df['FLUX_R']
        
        # create BGSBITS: bits associated to selection criteria
        if bgsbits:
            
            geocuts = getGeoCuts(df)
            photcuts = getPhotCuts(df)
            bgscuts = geocuts
            bgscuts.update(photcuts)
            
            BGSBITS = np.zeros_like(df['RA'], dtype='i8')
            BGSMASK = {}
            #[BGSMASK[key] = bit for bit, key in enumerate(bgscuts.keys())]
            
            print('---- BGSMASK key: ---- ')
            for bit, key in enumerate(bgscuts.keys()):
                
                BGSBITS |= bgscuts[key] * 2**(bit)
                BGSMASK[key] = bit
                print('\t %s, %i, %i' %(key, bit, 2**bit))
            # bgs selection
            #bgs = np.ones_like(df['RA'], dtype='?')
            #for key, val in zip(bgscuts.keys(), bgscuts.values()):
            #    if (key == 'allmask') or (key == 'MS'): continue
            #    else: bgs &= val
                    
            # dont forget the magnitude dependance
            #bgs &= flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R']) < 20
            bgs_any, bgs_bright, bgs_faint = get_bgs(df)
            bgs_sv_any, bgs_sv_bright, bgs_sv_faint, bgs_sv_faint_ext, bgs_sv_fibmag, bgs_sv_lowq = get_bgs_sv(df)
            
            BGSsel = {'bgs_any':bgs_any, 'bgs_bright':bgs_bright, 'bgs_faint':bgs_faint}
            BGSSVsel = {'bgs_sv_any':bgs_sv_any, 
                        'bgs_sv_bright':bgs_sv_bright, 
                        'bgs_sv_faint':bgs_sv_faint,
                        'bgs_sv_faint_ext':bgs_sv_faint_ext,
                        'bgs_sv_fibmag':bgs_sv_fibmag,
                        'bgs_sv_lowq':bgs_sv_lowq}
            
            for num, key in enumerate(BGSsel.keys()):
                BGSBITS |= BGSsel[key] * 2**(20+num)
                BGSMASK[key] = 20+num
                print('\t %s, %i, %i' %(key, 20+num, 2**(20+num)))
                
            for num, key in enumerate(BGSSVsel.keys()):
                BGSBITS |= BGSSVsel[key] * 2**(30+num)
                BGSMASK[key] = 30+num
                print('\t %s, %i, %i' %(key, 30+num, 2**(30+num)))
            
            tab['BGSBITS'] = BGSBITS
            
            # sanity check...
            print('---- Sanity Check ---- ')
            for bit, key in zip(BGSMASK.values(), BGSMASK.keys()):
                if key in list(BGSsel.keys()): print('\t %s, %i, %i' %(key, np.sum(BGSsel[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
                elif key in list(BGSSVsel.keys()): print('\t %s, %i, %i' %(key, np.sum(BGSSVsel[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
                else: print('\t %s, %i, %i' %(key, np.sum(bgscuts[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
                
            #print('\t %s, %i, %i' %('all', np.sum(bgs), np.sum(BGSBITS != 0)))
                
        print(sweepdir+sweep_file_name)
        np.save(sweepdir+sweep_file_name, tab)
    else:
        print('sweep file already exist at:%s' %(os.path.abspath(sweepdir+sweep_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    #get_area(patch, get_val = False)
    #print('Weight of %s catalogue: %s' %(sweep_file_name+'.npy', convert_size(os.path.getsize(sweep_file_name+'.npy'))))
    
    if not sweep_file: 
        if opt == '1': return tab
        if opt == '2': return dfsouth, dfnorth
    else: return np.load(os.path.abspath(sweepdir+sweep_file_name+'.npy'))
    
    
def cut_sweeps(patch=None, sweep_dir=None, rlimit=None, maskbitsource=False, opt='1'):
    '''Main function to extract the data from the SWEEPS'''
    
    j = 0
    sweepfiles = sorted(glob.glob(os.path.join(sweep_dir, '*.fits')))
    cols = ['RA', 'DEC', 'FLUX_R', 'FLUX_G', 'FLUX_Z', 'FIBERFLUX_R', 'MW_TRANSMISSION_R', 
                'MW_TRANSMISSION_G', 'MW_TRANSMISSION_Z','MASKBITS', 'REF_CAT', 'REF_ID', 
                    'GAIA_PHOT_G_MEAN_MAG', 'GAIA_ASTROMETRIC_EXCESS_NOISE', 'FRACFLUX_G', 
                        'FRACFLUX_R', 'FRACFLUX_Z', 'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z',
                             'FRACIN_G', 'FRACIN_R', 'FRACIN_Z', 'TYPE', 'FLUX_IVAR_R', 'FLUX_IVAR_G',
                                   'FLUX_IVAR_Z', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'SHAPEDEV_R', 'SHAPEDEV_R_IVAR',
                                       'SHAPEEXP_R', 'SHAPEEXP_R_IVAR']
    
    #sweepfiles = sweepfiles[:2]
    
    if opt == '1':
        #print('--------- OPTION 1 ---------')
        for i, file in enumerate(sweepfiles):
            
            if patch is not None:
                
                ramin, ramax, decmin, decmax = patch[0], patch[1], patch[2], patch[3]
                
                #cat1_path = cat1_paths[fileindex]
                #print(cat1_path)
                filename = file[-26:-5]
                brick = file[-20:-5]
                ra1min = float(brick[0:3])
                ra1max = float(brick[8:11])
                dec1min = float(brick[4:7])
                if brick[3]=='m':
                    dec1min = -dec1min
                dec1max = float(brick[-3:])
                if brick[-4]=='m':
                    dec1max = -dec1max
        
                r1=Rectangle(Point(ramin,decmin), Point(ramax, decmax), 'red')
                r2=Rectangle(Point(ra1min, dec1min), Point(ra1max, dec1max), 'blue')
        
                if not r1.intersects(r2):
                    continue
               
        
            if (i == 0) or (j == 0):
                cat = fitsio.read(file, columns=cols, upper=True, ext=1)
                if patch is not None: cat = cut(ramin, ramax, decmin, decmax, cat)
                keep = np.ones_like(cat, dtype='?')
                if rlimit != None:
                    rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                    keep &= rflux > 10**((22.5-rlimit)/2.5)
                
                if maskbitsource: 
                    keep &= (cat['REF_CAT'] != b'  ')
                
                print('\t fraction: %i/%i objects' %(np.sum(keep), len(cat)))
                cat0 = cat[keep]
                j += 1
                continue
        
            cat = fitsio.read(file, columns=cols, upper=True, ext=1)
            if patch is not None: cat = cut(ramin, ramax, decmin, decmax, cat)
            keep = np.ones_like(cat, dtype='?')
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                keep &= rflux > 10**((22.5-rlimit)/2.5)
                    
            if maskbitsource:
                keep &= (cat['REF_CAT'] != b'  ')
                        
            print('\t fraction: %i/%i objects' %(np.sum(keep), len(cat)))
        
            cat0 = np.concatenate((cat[keep], cat0))
            
            j += 1
            
        if j > 0: 
            print('\t Sample # objects: %i' %(len(cat0)))
        else: 
            cat0 = None
            print('\t Sample # objects: %i' %(0))
            
    if opt == '2':
        print('--------- OPTION 2 ---------')
        cat0 = {}
        for i in cols: cat0[i] = []
        for i, file in enumerate(sweepfiles):
            
            cat = fitsio.read(file, columns=cols, upper=True, ext=1)
            keep = np.ones_like(cat, dtype='?')
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                keep &= rflux > 10**((22.5-rlimit)/2.5)
                
            if maskbitsource:
                keep &= (cat['REF_CAT'] != b'  ')
                
            print('fraction: %i/%i objects' %(np.sum(keep), len(cat)))
            
            for j in cols:
                cat0[j] += cat[j][keep].tolist()
                
        
        print('Sample # objects: %i' %(len(cat0[list(cat0.keys())[0]])))
    
    return cat0

def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def bgsmask():
    
    mask = {'BS': 0,
            'MS': 1,
            'GC': 2,
            'LG': 3,
            'allmask': 4,
            'nobs': 5,
            'SG': 6,
            'SGSV': 7,
            'FMC': 8,
            'FMC2': 9,
            'CC': 10,
            'QC_FM': 11,
            'QC_FI': 12,
            'QC_FF': 13,
            'QC_IVAR': 14,
            'bgs_any': 20,
            'bgs_bright': 21,
            'bgs_faint': 22,
            'bgs_sv_any': 30,
            'bgs_sv_bright': 31,
            'bgs_sv_faint': 32,
            'bgs_sv_faint_ext': 33,
            'bgs_sv_fibmag': 34,
            'bgs_sv_lowq': 35
            }
    
    return mask

class Point:

    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord

class Rectangle:
    def __init__(self, bottom_left, top_right, colour):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.colour = colour

    def intersects(self, other):
        return not (self.top_right.x <= other.bottom_left.x or self.bottom_left.x >= other.top_right.x or self.top_right.y <= other.bottom_left.y or self.bottom_left.y >= other.top_right.y)
    
    def plot(self, other):
        fig, ax = plt.subplots(figsize=(15,8))
        rect = patches.Rectangle((self.bottom_left.x,self.bottom_left.y), abs(self.top_right.x - self.bottom_left.x), abs(self.top_right.y - self.bottom_left.y),linewidth=1.5, alpha=0.5, color='r')
        rect2 = patches.Rectangle((other.bottom_left.x,other.bottom_left.y), abs(other.top_right.x - other.bottom_left.x), abs(other.top_right.y - other.bottom_left.y),linewidth=1.5, alpha=0.5, color='blue')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        xlims = np.array([self.bottom_left.x, self.top_right.x, other.bottom_left.x, other.top_right.x])
        ylims = np.array([self.bottom_left.y, self.top_right.y, other.bottom_left.y, other.top_right.y])
        ax.set_xlim(xlims.min()-1, xlims.max()+1)
        ax.set_ylim(ylims.min()-1, ylims.max()+1)
        
def cut(ramin, ramax, decmin, decmax, catalog):
    
    mask = np.logical_and(catalog['RA'] >= ramin, catalog['RA'] <= ramax)
    mask &= np.logical_and(catalog['DEC'] >= decmin, catalog['DEC'] <= decmax)
    cat = catalog[mask]
    
    return cat

def get_dict(cat=None, randoms=None, pixmapfile=None, hppix_ran=None, hppix_cat=None, maskrand=None, maskcat=None, 
                 getnobs=False, nside=None, npix=None, nest=None, pixarea=None, Nranfiles=None, 
                     ranindesi=None, catindesi=None, dec_resol_ns=32.375, namesels=None, galb=None, survey='main', 
                         desifootprint=True, target_outputs=True, log=False, tiledir='/global/cscratch1/sd/raichoor/', ws=None):

   # start = raichoorlib.get_date()
    # creating dictionary
    hpdict = {}
    
    if (nside is None) or (npix is None) or (nest is None) or (pixarea is None) & (pixmapfile is not None):
        
        hdr          = fits.getheader(pixmapfile,1)
        nside,nest   = hdr['hpxnside'],hdr['hpxnest']
        npix         = hp.nside2npix(nside)
        pixarea      = hp.nside2pixarea(nside,degrees=True)
    else: raise ValueError('if not pixel information given, include pixmapfile to compute them.')
    
    if (getnobs) and (randoms is None):
        raise ValueError('random catalogue can not be None when getnobs is set True.')
        
    if (hppix_ran is None) and (randoms is not None):
        hppix_ran = hp.ang2pix(nside,(90.-np.array(randoms['DEC']))*np.pi/180.,np.array(randoms['RA'])*np.pi/180.,nest=nest)
    elif (hppix_ran is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute their hp pixels indexes.')
        
    if (ranindesi is None) and (randoms is not None):
        ranindesi = get_isdesi(randoms['RA'],randoms['DEC'], tiledir=tiledir) # True if is in desi footprint
    elif (ranindesi is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute ranindesi.')
        
        
    theta,phi  = hp.pix2ang(nside,np.arange(npix),nest=nest)
    hpdict['ra'],hpdict['dec'] = 180./np.pi*phi,90.-180./np.pi*theta
    c = SkyCoord(hpdict['ra']*units.degree,hpdict['dec']*units.degree, frame='icrs')
    hpdict['gall'],hpdict['galb'] = c.galactic.l.value,c.galactic.b.value

    # is in desi tile?
    hpdict['isdesi'] = get_isdesi(hpdict['ra'],hpdict['dec'], tiledir=tiledir)
    if log: print('positions and desifotprint DONE...')

    # propagating some keys from ADM pixweight
    hdu = fits.open(pixmapfile)
    data = hdu[1].data
    for key in ['HPXPIXEL', 'FRACAREA', 
            'STARDENS', 'EBV', 
            'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
            'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
            'PSFDEPTH_W1', 'PSFDEPTH_W2',
            'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']:
        if (key=='STARDENS'):
            hpdict['log10_'+key.lower()] = np.log10(data[key])
            hpdict[key.lower()] = data[key]
        elif (key[:8]=='GALDEPTH'):
            hpdict[key.lower()] = 22.5-2.5*np.log10(5./np.sqrt(data[key]))
        else:
            hpdict[key.lower()] = data[key]
    if log: print('systematics DONE...')
        
    # computing fracareas
    randdens = 5000*Nranfiles
    if log: print('randdens = ', randdens, ' ; len randoms = ', len(hppix_ran))
    if desifootprint: mainfootprint = ranindesi
    else: mainfootprint = np.ones_like(hppix_ran, dtype='?')
    if maskrand is None:
        ind,c           = np.unique(hppix_ran[mainfootprint],return_counts=True)
    else:
        ind,c           = np.unique(hppix_ran[(maskrand) & (mainfootprint)],return_counts=True)
    hpdict['bgsfracarea']      = np.zeros(npix)
    hpdict['bgsfracarea'][ind] = c / randdens / pixarea
    if log: print('bgsfracarea DONE...')
    
    # computing nobs
    if getnobs:
        import pandas as pd
        if maskrand is None:
            s = pd.Series(hppix_ran[mainfootprint])
        else:
            s = pd.Series(hppix_ran[(maskrand) & (mainfootprint)])
        d = s.groupby(s).groups
        for i in ['NOBS_G', 'NOBS_R', 'NOBS_Z']:
            hpdict[i] = np.zeros(npix)
            for j in d.keys():
                hpdict[i][j] = np.mean(randoms[i][d[j]])
        if log: print('nobs DONE...')
        
    # north/south/des/decals
    hpdict['issouth'] = np.zeros(npix,dtype=bool)
    tmp               = (hpdict['bgsfracarea']>0) & ((hpdict['galb']<0) | ((hpdict['galb']>0) & (hpdict['dec']<dec_resol_ns)))
    hpdict['issouth'][tmp] = True
    hpdict['isnorth'] = np.zeros(npix,dtype=bool)
    tmp               = (hpdict['bgsfracarea']>0) & (hpdict['dec']>dec_resol_ns) & (hpdict['galb']>0)
    hpdict['isnorth'][tmp] = True
    hpdict['isdes']   = get_isdes(hpdict['ra'],hpdict['dec'])
    hpdict['isdecals'] = (hpdict['issouth']) & (~hpdict['isdes'])
    hpdict['issouth_n'] = (hpdict['issouth']) & (hpdict['galb']>0)
    hpdict['issouth_s'] = (hpdict['issouth']) & (hpdict['galb']<0)
    
    hpdict['issvfields'] = get_svfields(hpdict['ra'],hpdict['dec'])
    hpdict['issvfields_n'] = (hpdict['issvfields']) & (hpdict['isnorth'])
    hpdict['issvfields_s'] = (hpdict['issvfields']) & (hpdict['issouth'])
    
    hpdict['issvfields_fg'] = get_svfields_fg(hpdict['ra'],hpdict['dec'])
    hpdict['issvfields_fg_n'] = (hpdict['issvfields_fg']) & (hpdict['isnorth'])
    hpdict['issvfields_fg_s'] = (hpdict['issvfields_fg']) & (hpdict['issouth'])
    
    hpdict['issvfields_ij'] = get_svfields_ij(hpdict['ra'],hpdict['dec'], survey='all')
    hpdict['issvfields_ij_n'] = (hpdict['issvfields_ij']) & (hpdict['isnorth'])
    hpdict['issvfields_ij_s'] = (hpdict['issvfields_ij']) & (hpdict['issouth'])
    
    regs = ['south','decals','des','north', 'south_n', 'south_s', 'svfields', 'svfields_n', 'svfields_s', 
           'svfields_fg', 'svfields_fg_n', 'svfields_fg_s', 'svfields_ij', 'svfields_ij_n', 'svfields_ij_s']
    
    #hpdict['istest'] = (hpdict['ra'] > 160.) & (hpdict['ra'] < 230.) & (hpdict['dec'] > -2.) & (hpdict['dec'] < 18.)
    if log: print('regions DONE...')

    # areas
    hpdict['area_all']   = hpdict['bgsfracarea'].sum() * pixarea
    if log: print('area_'+'all'+' = '+'%.0f'%hpdict['area_'+'all']+' deg2')
    for reg in regs:
        hpdict['bgsarea_'+reg]   = hpdict['bgsfracarea'][hpdict['is'+reg]].sum() * pixarea
        if log: print('bgsarea_'+reg+' = '+'%.0f'%hpdict['bgsarea_'+reg]+' deg2')
    if log: print('areas DONE...')
    
    
    #target densities
    if target_outputs:
        
        if (cat is None) or (pixmapfile is None) or (namesels is None) or (Nranfiles is None):
            raise ValueError('cat, pixmapfile, namesels and Nranfiles can not be None.')
        
        if (hppix_cat is None):
            hppix_cat = hp.ang2pix(nside,(90.-cat['DEC'])*np.pi/180.,cat['RA']*np.pi/180.,nest=nest) # catalogue hp pixels array
        
        if (catindesi is None) & (desifootprint):
            catindesi = get_isdesi(cat['RA'],cat['DEC'], tiledir=tiledir) # True is is in desi footprint
        
        if galb is None:
            c = SkyCoord(cat['RA']*units.degree,cat['DEC']*units.degree, frame='icrs')
            galb = c.galactic.b.value # galb coordinate
        
        #namesels = {'any':-1, 'bright':1, 'faint':0, 'wise':2}
        for foot in ['north','south']:
        
            data = cat
        
            if (foot=='north'): keep = (data['DEC']>dec_resol_ns) & (galb>0)
            if (foot=='south'): keep = (data['DEC']<dec_resol_ns) | (galb<0)        
            ##
            if desifootprint:
                if maskcat is None: keep &= catindesi
                else: keep &= (maskcat) & (catindesi)
            else:
                if maskcat is not None: keep &= maskcat
        
            if survey == 'main': bgstargetname = 'BGS_TARGET'
            elif survey == 'sv1': bgstargetname = 'SV1_BGS_TARGET'
            elif survey == 'bgs': bgstargetname = 'BGSBITS'

            for namesel, bitnum in zip(namesels.keys(), namesels.values()):
                if log: print('computing for ', foot, '/', namesel)
                    
                if survey == 'custom':
                    sel = bitnum
                else:
                    if (namesel=='any'):             sel = np.ones(len(data),dtype=bool)
                    else:                            sel = ((data[bgstargetname] & 2**(bitnum)) != 0)
            
                ind,c = np.unique(hppix_cat[(sel) & (keep)],return_counts=True)
                hpdict[foot+'_n'+namesel]      = np.zeros(npix)
                hpdict[foot+'_n'+namesel][ind] = c
                
                if ws is not None: hpdict[foot+'_n'+namesel] = hpdict[foot+'_n'+namesel]*ws
                
            if log: print('target densities in %s DONE...' %(foot))
            
        # storing mean hpdens
        if desifootprint: isdesi = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
        else: isdesi = (hpdict['bgsfracarea']>0)
        for namesel in namesels.keys():
            ## south + north density
            hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
            ## split per region
            hpdict['meandens_'+namesel+'_'+'all'] = np.nanmean(hpdens[isdesi])
            if log: print('meandens_'+namesel+'_'+'all'+' = '+'%.0f'%hpdict['meandens_'+namesel+'_'+'all']+' /deg2')
            for reg in regs:
                hpdict['meandens_'+namesel+'_'+reg] = np.nanmean(hpdens[(isdesi) & (hpdict['is'+reg])])
                if log: print('meandens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+' /deg2')
            
        # storing total target density
        #isdesi = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
        for namesel in namesels.keys():
            ## split per region
            hpdict['dens_'+namesel+'_'+'all'] = (hpdict['south_n'+namesel][isdesi] + hpdict['north_n'+namesel][isdesi]).sum() / (pixarea * hpdict['bgsfracarea'][isdesi].sum())
            if log: print('dens_'+namesel+'_'+'all'+' = '+'%.0f'%hpdict['dens_'+namesel+'_'+'all']+' /deg2')
            for reg in regs:
                hpdict['dens_'+namesel+'_'+reg] = (hpdict['south_n'+namesel][(isdesi) & (hpdict['is'+reg])] + hpdict['north_n'+namesel][(isdesi) & (hpdict['is'+reg])]).sum() / (pixarea * hpdict['bgsfracarea'][(isdesi) & (hpdict['is'+reg])].sum())
                if log: print('dens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['dens_'+namesel+'_'+reg]+' /deg2')
    
    return hpdict

# is in desi nominal footprint? (using tile radius of 1.6 degree)
# small test shows that it broadly works to cut on desi footprint 
def get_isdesi(ra,dec, nest=True, tiledir='/global/cscratch1/sd/raichoor/'):
    radius   = 1.6 # degree
    tmpnside = 16
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open(tiledir+'desi-tiles-viewer.fits')
    data = hdu[1].data
    keep = (data['in_desi']==1)
    data = data[keep]
    tra,tdec = data['ra'],data['dec']
    # get hppix inside desi tiles
    theta,phi  = hp.pix2ang(tmpnside,np.arange(tmpnpix),nest=nest)
    hpra,hpdec = 180./np.pi*phi,90.-180./np.pi*theta
    hpindesi   = np.zeros(tmpnpix,dtype=bool)
    _,ind,_,_,_= search_around(tra,tdec,hpra,hpdec,search_radius=1.6*3600, verbose=False)
    hpindesi[np.unique(ind)] = True
    ## small hack to recover few rejected pixels inside desi. Avoid holes if any
    tmp  = np.array([i for i in range(tmpnpix) 
                     if hpindesi[hp.get_all_neighbours(tmpnside,i,nest=nest)].sum()==8])
    hpindesi[tmp] = True
    ##
    pixkeep    = np.where(hpindesi)[0]
    # now compute the hppix for the tested positions
    pix  = hp.ang2pix(tmpnside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    keep = np.in1d(pix,pixkeep)
    return keep

def get_isbgstile(ra,dec):
    radius   = 1.6 # degree
    tmpnside = 256
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open('/global/cscratch1/sd/qmxp55/BGS_SV_30_3x_superset60_JUL2019.fits')
    data = hdu[1].data
    #keep = (data['in_desi']==1)
    #data = data[keep]
    tra,tdec = data['RA'],data['DEC']
    # get hppix inside desi tiles
    theta,phi  = hp.pix2ang(tmpnside,np.arange(tmpnpix),nest=nest)
    hpra,hpdec = 180./np.pi*phi,90.-180./np.pi*theta
    hpindesi   = np.zeros(tmpnpix,dtype=bool)
    
    idx,ind,_,_,_= search_around(tra,tdec,hpra,hpdec,search_radius=1.6*3600)
    
    hpindesi[np.unique(ind)] = True
    ## small hack to recover few rejected pixels inside desi. Avoid holes if any
    #tmp  = np.array([i for i in range(tmpnpix) 
    #                 if hpindesi[hp.get_all_neighbours(tmpnside,i,nest=nest)].sum()==8])
    #hpindesi[tmp] = True
    ##
    pixkeep    = np.where(hpindesi)[0]
    # now compute the hppix for the tested positions
    pix  = hp.ang2pix(tmpnside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    keep = np.in1d(pix,pixkeep)
    
    hptileid = np.ones(tmpnpix)*float('NaN')
    tileid = np.ones_like(pix)*float('NaN')
    for i in range(len(data)):
        mask = ind[idx == i]
        #print(i, len(mask), len(np.unique(mask)))
        hptileid[np.unique(mask)] = data['CENTERID'][i] #hp tile center id
        mask2 = np.in1d(pix,mask)
        tileid[np.where(mask2)] = data['CENTERID'][i] #cat tile center id
    
    return keep, tileid

def get_random(N=3, sweepsize=None, dr='dr8', dirpath='/global/cscratch1/sd/qmxp55/'):
    
    import time
    start = time.time()
        
    if (N < 2) & (dr == 'dr8'):
        raise ValueError('Number of RANDOMS files must be greater than one')
    
    import glob
    #ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.29.0/' #issues with MASKBITS...
    
    if dr == 'dr8-south': dr = 'dr8'
    random_file_name = '%s_random_N%s' %(dr, str(N))
        
    random_file = os.path.isfile(dirpath+random_file_name+'.npy')
    if not random_file:
        if dr is 'dr7':
            ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.22.0/'
            randoms = glob.glob(ranpath + 'randoms*')
        elif (dr == 'dr8'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr8/0.31.0/randoms/'
            randoms = glob.glob(ranpath + 'randoms-inside*')
        elif (dr == 'dr9sv'):
            ranpath = '/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/randoms/'
            randoms = [ranpath + 'randoms-dr9-hp-X-1.fits']
        elif (dr == 'dr9f'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr9f/0.38.0/randoms/resolve/'
            randoms = [ranpath + 'randoms-dr9-hp-X-1.fits']
        elif (dr == 'dr9g'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr9g/0.38.0/randoms/resolve/'
            randoms = [ranpath + 'randoms-dr9-hp-X-1.fits']
        elif (dr == 'dr9i'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr9i/0.40.0/randoms/resolve/'
            randoms = [ranpath + 'randoms-dr9-hp-X-1.fits']
        elif (dr == 'dr9j'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr9j/0.40.0/randoms/resolve/'
            randoms = [ranpath + 'randoms-dr9-hp-X-1.fits']
        elif (dr == 'dr9d'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr9d/PRnone/randoms/'
            random_south = ranpath + 'dr9d-south/' + 'randoms-dr9-hp-X-1.fits'
            random_north = ranpath + 'dr9d-north/' + 'randoms-dr9-hp-X-1.fits'
            randoms = [random_south, random_north]
            
        if len(randoms) > 1: randoms.sort()
        if dr == 'dr8': randoms = randoms[0:N]

        for i in range(len(randoms)):
            df_ran = fitsio.read(randoms[i], columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
        
            if i == 0:
                df_ranS1 = df_ran
                continue
        
            df_ranS1 = np.concatenate((df_ranS1, df_ran))
            
        np.save(dirpath+random_file_name, df_ranS1)
            
        print('# objects in RANDOM: %i' %(len(df_ranS1)))
        if sweepsize is not None:
            print('fraction of RANDOM catalogue compared to SWEEP catalogue: %2.3g' 
                      %(len(df_ranS1)/sweepsize))
    else:
        print('RANDOM file already exist at:%s' %(os.path.abspath(dirpath+random_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    print('Weight of %s catalogue: %s' %(dirpath+random_file_name+'.npy', convert_size(os.path.getsize(dirpath+random_file_name+'.npy'))))
    
    if not random_file:
        return df_ranS1
    else:
        return np.load(dirpath+random_file_name+'.npy')
    
def convert_size(size_bytes): 
    import math
    if size_bytes == 0: 
            return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "%s %s" % (size, size_name[i])


# https://desi.lbl.gov/svn/docs/technotes/targeting/target-truth/trunk/python/match_coord.py
# slightly edited (plot_q and keep_all_pairs removed; u => units)
def match_coord(ra1, dec1, ra2, dec2, search_radius=1., nthneighbor=1, verbose=True):
	'''
	Match objects in (ra2, dec2) to (ra1, dec1). 

	Inputs: 
		RA and Dec of two catalogs;
		search_radius: in arcsec;
		(Optional) keep_all_pairs: if true, then all matched pairs are kept; otherwise, if more than
		one object in t2 is match to the same object in t1 (i.e. double match), only the closest pair
		is kept.

	Outputs: 
		idx1, idx2: indices of matched objects in the two catalogs;
		d2d: distances (in arcsec);
		d_ra, d_dec: the differences (in arcsec) in RA and Dec; note that d_ra is the actual angular 
		separation;
	'''
	t1 = Table()
	t2 = Table()
	# protect the global variables from being changed by np.sort
	ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])
	t1['ra'] = ra1
	t2['ra'] = ra2
	t1['dec'] = dec1
	t2['dec'] = dec2
	t1['id'] = np.arange(len(t1))
	t2['id'] = np.arange(len(t2))
	# Matching catalogs
	sky1 = SkyCoord(ra1*units.degree,dec1*units.degree, frame='icrs')
	sky2 = SkyCoord(ra2*units.degree,dec2*units.degree, frame='icrs')
	idx, d2d, d3d = sky2.match_to_catalog_sky(sky1, nthneighbor=nthneighbor)
	# This finds a match for each object in t2. Not all objects in t1 catalog are included in the result. 

	# convert distances to numpy array in arcsec
	d2d = np.array(d2d.to(units.arcsec))
	matchlist = d2d<search_radius
	if np.sum(matchlist)==0:
		if verbose:
			print('0 matches')
		return np.array([], dtype=int), np.array([], dtype=int), np.array([]), np.array([]), np.array([])
	t2['idx'] = idx
	t2['d2d'] = d2d
	t2 = t2[matchlist]
	init_count = np.sum(matchlist)
	#--------------------------------removing doubly matched objects--------------------------------
	# if more than one object in t2 is matched to the same object in t1, keep only the closest match
	t2.sort('idx')
	i = 0
	while i<=len(t2)-2:
		if t2['idx'][i]>=0 and t2['idx'][i]==t2['idx'][i+1]:
			end = i+1
			while end+1<=len(t2)-1 and t2['idx'][i]==t2['idx'][end+1]:
				end = end+1
			findmin = np.argmin(t2['d2d'][i:end+1])
			for j in range(i,end+1):
				if j!=i+findmin:
					t2['idx'][j]=-99
			i = end+1
		else:
			i = i+1

	mask_match = t2['idx']>=0
	t2 = t2[mask_match]
	t2.sort('id')
	if verbose:
		print('Doubly matched objects = %d'%(init_count-len(t2)))
	# -----------------------------------------------------------------------------------------
	if verbose:
		print('Final matched objects = %d'%len(t2))
	# This rearranges t1 to match t2 by index.
	t1 = t1[t2['idx']]
	d_ra = (t2['ra']-t1['ra']) * 3600.    # in arcsec
	d_dec = (t2['dec']-t1['dec']) * 3600. # in arcsec
	##### Convert d_ra to actual arcsecs #####
	mask = d_ra > 180*3600
	d_ra[mask] = d_ra[mask] - 360.*3600
	mask = d_ra < -180*3600
	d_ra[mask] = d_ra[mask] + 360.*3600
	d_ra = d_ra * np.cos(t1['dec']/180*np.pi)
	##########################################
	return np.array(t1['id']), np.array(t2['id']), np.array(t2['d2d']), np.array(d_ra), np.array(d_dec)

# copied from https://github.com/rongpu/desi-examples/blob/master/bright_star_contamination/match_coord.py
def search_around(ra1, dec1, ra2, dec2, search_radius=1., verbose=True):

    ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])

    sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    sky2 = SkyCoord(ra2*units.degree,dec2*units.degree, frame='icrs')
    idx1, idx2, d2d, d3d = sky2.search_around_sky(sky1, seplimit=search_radius*units.arcsec)

    if verbose:
        print('%d nearby objects'%len(idx1))
    # convert distances to numpy array in arcsec
    d2d   = np.array(d2d.to(units.arcsec))
    d_ra  = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
    d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask       = d_ra > 180*3600
    d_ra[mask] = d_ra[mask] - 360.*3600
    mask       = d_ra < -180*3600
    d_ra[mask] = d_ra[mask] + 360.*3600
    d_ra       = d_ra * np.cos(dec1[idx1]/180*np.pi)
    ##########################################

    return idx1, idx2, d2d, d_ra, d_dec


def get_isdes(ra,dec):
	hdu = fits.open('/global/cscratch1/sd/raichoor/desits/des_hpmask.fits')
	nside,nest = hdu[1].header['HPXNSIDE'],hdu[1].header['HPXNEST']
	hppix     = hp.ang2pix(nside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
	isdes     = np.zeros(len(ra),dtype=bool)
	isdes[np.in1d(hppix,hdu[1].data['hppix'])] = True
	return isdes

def get_reg(reg='decals', hppix=None):
    ''' get an specific LS region i.e., DECaLS/DES/NORTH from catalogue with hppix index info.'''
    
    hpdict = np.load('/global/cscratch1/sd/qmxp55/hppixels_regions.npy')
    isreg_pixlist = hpdict['hpxpixel'][hpdict['is'+reg]]
    regcat = np.in1d(hppix, isreg_pixlist)
    
    return regcat

def get_svfields(ra, dec):
    
    svfields = {}

    svfields['s82'] = [30, 40, -7, 2.0] #90
    svfields['egs'] = [210, 220, 50, 55] #31
    svfields['g09'] = [129, 141, -2.0, 3.0] #60
    svfields['g12'] = [174, 186, -3.0, 2.0] #60
    svfields['g15'] = [211, 224, -2.0, 3.0] #65
    svfields['overlap'] = [135, 160, 30, 35] #104
    svfields['refnorth'] = [215, 230, 41, 46] #56
    svfields['ages'] = [215, 220, 30, 40] #40
    svfields['sagittarius'] = [200, 210, 5, 10] #49
    svfields['highebv_n'] = [140, 150, 65, 70] #19
    svfields['highebv_s'] = [240, 245, 20, 25] #23
    svfields['highstardens_n'] = [273, 283, 40, 45] #37
    svfields['highstardens_s'] = [260, 270, 15, 20] #47
    svfields['s82_s'] = [330, 340, -2, 3] #51
    
    
    keep = np.zeros_like(ra, dtype='?')
    for key, val in zip(svfields.keys(), svfields.values()):
        keep |= ((ra > val[0]) & (ra < val[1]) & (dec > val[2]) & (dec < val[3]))
        
    return keep

def get_svfields_fg(ra, dec):
    
    svfields = {}

    #svfields['s82'] = [30, 40, -7, 2.0] #90
    #svfields['egs'] = [210, 220, 50, 55] #31
    #svfields['g09'] = [129, 141, -2.0, 3.0] #60
    #svfields['g12'] = [174, 186, -3.0, 2.0] #60
    svfields['g15'] = [211, 224, -2.0, 3.0] #65
    #svfields['overlap'] = [135, 160, 30, 35] #104
    svfields['refnorth'] = [215, 230, 41, 46] #56
    svfields['ages'] = [215, 220, 30, 40] #40
    svfields['sagittarius'] = [200, 210, 5, 10] #49
    #svfields['highebv_n'] = [140, 150, 65, 70] #19
    svfields['highebv_s'] = [240, 245, 20, 25] #23
    svfields['highstardens_n'] = [273, 283, 40, 45] #37
    svfields['highstardens_s'] = [260, 270, 15, 20] #47
    #svfields['s82_s'] = [330, 340, -2, 3] #51
    
    
    keep = np.zeros_like(ra, dtype='?')
    for key, val in zip(svfields.keys(), svfields.values()):
        keep |= ((ra > val[0]) & (ra < val[1]) & (dec > val[2]) & (dec < val[3]))
        
    return keep

def get_svfields_ij(ra, dec, survey='all'):
    
    if survey == 'south':
        svfields = {}
        svfields['g15'] = [211, 224, -2.0, 3.0] #65
     
    if survey == 'north':
        svfields = {}
        svfields['refnorth'] = [215.2, 229.8, 41, 46] #56
        
    if survey == 'all':
        svfields = {}
        svfields['refnorth'] = [215.2, 229.8, 41, 46] #56
        svfields['g15'] = [211, 224, -2.0, 3.0] #65

    
    keep = np.zeros_like(ra, dtype='?')
    for key, val in zip(svfields.keys(), svfields.values()):
        keep |= ((ra > val[0]) & (ra < val[1]) & (dec > val[2]) & (dec < val[3]))
        
    return keep

def get_msmask(masksources):
    
    mag = np.zeros_like(masksources['RA'])
    ingaia = (masksources['REF_CAT'] == b'G2') & (masksources['G'] <= 16)
    intycho = (masksources['REF_CAT'] == b'T2')
    
    # get MAG_VT mag from Tycho
    path = '/global/homes/q/qmxp55/DESI/matches/'
    tycho = fitsio.read(path+'tycho2.fits')
    idx2, idx1, d2d, d_ra, d_dec = search_around(masksources['RA'][intycho], masksources['DEC'][intycho], tycho['RA'], tycho['DEC'], search_radius=0.2)
    mag[intycho] = tycho['MAG_VT'][idx1]
    
    mag[np.where(ingaia)] = masksources['G'][ingaia]
    keep = (ingaia) | (intycho)
    
    tab = Table()
    for col in ['RA', 'DEC', 'MAG', 'REF_CAT']:
        if col == 'MAG': tab[col] = mag[keep]
        else: tab[col] = masksources[col][keep]
    print('%i Medium Bright Stars' %(np.sum(keep)))
    
    return tab

def get_bsmask(masksources):
    
    mag = np.zeros_like(masksources['RA'])
    ingaia = (masksources['REF_CAT'] == b'G2') & (masksources['G'] <= 13)
    intycho = (masksources['REF_CAT'] == b'T2')
    
    # get MAG_VT mag from Tycho
    path = '/global/homes/q/qmxp55/DESI/matches/'
    tycho = fitsio.read(path+'tycho2.fits')
    idx2, idx1, d2d, d_ra, d_dec = search_around(masksources['RA'][intycho], masksources['DEC'][intycho], tycho['RA'], tycho['DEC'], search_radius=0.2)
    mag[intycho] = tycho['MAG_VT'][idx1]
    
    mag[np.where(ingaia)] = masksources['G'][ingaia]
    keep = (ingaia) | (intycho)
    
    tab = Table()
    for col in ['RA', 'DEC', 'MAG', 'REF_CAT']:
        if col == 'MAG': tab[col] = mag[keep]
        else: tab[col] = masksources[col][keep]
    print('%i Bright Stars' %(np.sum(keep)))
    
    return tab

def gaiaAEN(inGAIA=None, size=None, G=None, AEN=None, dr='dr8'):
    #definition for DR8 
    
    stars_aen = np.zeros(size, dtype='?')
    if dr == 'dr8':
        #Stars with AEN class...
        stars_aen |= ((inGAIA) & (G < 19) & (AEN < 10**(0.5))) 
        stars_aen |= ((inGAIA) & (G >= 19) & (AEN < 10**(0.5 + 0.2*(G - 19.))))
    elif (dr[:3] == 'dr9') or (dr == 'dr9f') or (dr == 'dr9g'):
        #Stars with AEN class...
        stars_aen |= ((inGAIA) & (G < 18) & (AEN < 10**(0.5)))
    else:
        raise ValueError('%s is not a valid data release.' %(dr))
    
    #Galaxies with AEN class...
    gal_aen = (inGAIA) & (~stars_aen)
    
    return stars_aen, gal_aen

def query_catalog_mask(ra, dec, starCat, radii, nameMag='MAG_VT', diff_spikes=True, length_radii=None, widht_radii=None, return_diagnostics=False, bestfit=True, log=False):
    '''
    Catalog-based WISE bright star mask.
    Input:
    ra, dec: coordinates;
    diff_spikes: apply diffraction spikes masking if True;
    return_diagnostics: return disgnostic information if True;
    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    import time
    from QA import circular_mask_radii_func
    
    start = time.time()
    
    wisecat = starCat

    w1_ab = np.array(wisecat[nameMag])
    raW = np.array(wisecat['RA'])
    decW = np.array(wisecat['DEC'])

    w1_bins = np.arange(0, 22, 0.5)
    
    # only flagged by the circular mask (True if contaminated):
    circ_flag = np.zeros(len(ra), dtype=bool)
    # flagged by the diffraction spike mask but not the circular mask (True if contaminated):
    ds_flag = np.zeros(len(ra), dtype=bool)
    # flagged in the combined masks (True if contaminated):
    cat_flag = np.zeros(len(ra), dtype=bool)

    # record the magnitude of the star that causes the contamination and distance to it
    w1_source = np.zeros(len(ra), dtype=float)
    d2d_source = np.zeros(len(ra), dtype=float)

    ra2, dec2 = map(np.copy, [ra, dec])
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

    for index in range(len(w1_bins)-1):

        mask_wise = (w1_ab>=w1_bins[index]) & (w1_ab<=w1_bins[index+1])
        if log: print('{:.2f} < {} < {:.2f}   {} TYCHO bright stars'.format(w1_bins[index], nameMag, w1_bins[index+1], np.sum(mask_wise)))

        if np.sum(mask_wise)==0:
            continue
    
        # find the maximum mask radius for the magnitude bin        
        if not diff_spikes:
            search_radius = np.max(circular_mask_radii_func(w1_ab[mask_wise], radii, bestfit=bestfit))
        else:
            # Define length for diffraction spikes mask
            x, y = np.transpose(length_radii)
            ds_mask_length_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))
            search_radius = np.max([circular_mask_radii_func(w1_ab[mask_wise], radii, bestfit=bestfit), 0.5*ds_mask_length_func(w1_ab[mask_wise])])

        # Find all pairs within the search radius
        ra1, dec1 = map(np.copy, [raW[mask_wise], decW[mask_wise]])
        sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
        idx_wise, idx_decals, d2d, _ = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
        if log: print('%d nearby objects'%len(idx_wise))
        
        # convert distances to numpy array in arcsec
        d2d = np.array(d2d.to(u.arcsec))

        d_ra = (ra2[idx_decals]-ra1[idx_wise])*3600.    # in arcsec
        d_dec = (dec2[idx_decals]-dec1[idx_wise])*3600. # in arcsec
        ##### Convert d_ra to actual arcsecs #####
        mask = d_ra > 180*3600
        d_ra[mask] = d_ra[mask] - 360.*3600
        mask = d_ra < -180*3600
        d_ra[mask] = d_ra[mask] + 360.*3600
        d_ra = d_ra * np.cos(dec1[idx_wise]/180*np.pi)
        ##########################################

        # circular mask
        mask_radii = circular_mask_radii_func(w1_ab[mask_wise][idx_wise], radii, bestfit=bestfit)
        # True means contaminated:
        circ_contam = d2d < mask_radii
        circ_flag[idx_decals[circ_contam]] = True

        w1_source[idx_decals[circ_contam]] = w1_ab[mask_wise][idx_wise[circ_contam]]
        d2d_source[idx_decals[circ_contam]] = d2d[circ_contam]

        if diff_spikes:

            ds_contam = ds_masking_func(d_ra, d_dec, d2d, w1_ab[mask_wise][idx_wise], length_radii, widht_radii)
            ds_flag[idx_decals[ds_contam]] = True

            # combine the two masks
            cat_flag[idx_decals[circ_contam | ds_contam]] = True

            w1_source[idx_decals[ds_contam]] = w1_ab[mask_wise][idx_wise[ds_contam]]
            d2d_source[idx_decals[ds_contam]] = d2d[ds_contam]

            if log: print('{} objects masked by circular mask'.format(np.sum(circ_contam)))
            if log: print('{} additionally objects masked by diffraction spikes mask'.format(np.sum(circ_contam | ds_contam)-np.sum(circ_contam)))
            if log: print('{} objects masked by the combined masks'.format(np.sum(circ_contam | ds_contam)))
            if log: print()

        else:

            if log: print('{} objects masked'.format(np.sum(circ_contam)))
            if log: print()

    if not diff_spikes:
        cat_flag = circ_flag
        
    end = time.time()
    print('Total run time: %f sec' %(end - start))

    if not return_diagnostics:
        return cat_flag
    else:
        # package all the extra info
        more_info = {}
        more_info['w1_source'] = w1_source
        more_info['d2d_source'] = d2d_source
        more_info['circ_flag'] = circ_flag
        more_info['ds_flag'] = ds_flag
        
    
    return cat_flag, more_info

def LSLGA_fit(LSLGA, radii=None, N=None):
    
    from QA import circular_mask_radii_func
    
    MAG = np.array(LSLGA['MAG'])
    if radii == 'mag-rad':
        major = circular_mask_radii_func(MAG, radii)/3600.#[degrees]
    elif radii == 'D25':
        major = N*LSLGA['D25']/2./60. #[degrees]
    else:
        raise ValueError('User a valid radii:{D25,mag-rad}')
        
    minor = major*LSLGA['BA']#[degrees]
    angle = 90 - LSLGA['PA']
    
    return LSLGA['RA'], LSLGA['DEC'], major, minor, angle

def LSLGA_veto(cat=None, LSLGA=None, radii='D25', N=1):
    
    import time
    start = time.time()
    
    RA, DEC, major, minor, angle = LSLGA_fit(LSLGA, radii, N)
    centers = (RA, DEC)
    mask = veto_ellip((cat['RA'], cat['DEC']), centers, major, minor, angle)
    
    end = time.time()
    print('Total run time: %f sec' %(end - start))
    
    return mask
