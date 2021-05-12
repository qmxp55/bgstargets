#
import numpy as np

def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def addBackLG(df, randoms=False, survey='main'):
    
    if not randoms:
        refcat = df['REF_CAT'] 
        #the LSLGA galaxies
        #L2 = ((refcat == 'L2') | (refcat == b'L2') | (refcat == 'L2 ') | (refcat == b'L2 '))
        if survey == 'main':
            if isinstance(np.atleast_1d(refcat)[0], str):
                LX = [(rc[0] == "L") if len(rc) > 0 else False for rc in refcat]
            else:
                LX = [(rc.decode()[0] == "L") if len(rc) > 0 else False for rc in refcat]
            LX = np.array(LX, dtype=bool)
            BS = (np.uint64(df['MASKBITS']) & np.uint64(2**1))==0
            GC = (np.uint64(df['MASKBITS']) & np.uint64(2**13))==0
            LX &= (BS) & (GC)
        elif survey == 'sv':
            LX = np.zeros_like(df['RA'], dtype=bool)
        else: raise ValueError('%s is not a valid survey programme.' %(survey))
            
    else:
        LX = np.zeros_like(df['RA'], dtype=bool)
        
    return LX
    
def getGeoCuts(df, randoms=False, survey='main'):
    
    #GEO cuts CATALOGUE
    BS = (np.uint64(df['MASKBITS']) & np.uint64(2**1))==0
    MS = (np.uint64(df['MASKBITS']) & np.uint64(2**11))==0
    GC = (np.uint64(df['MASKBITS']) & np.uint64(2**13))==0
    LG = ((np.uint64(df['MASKBITS']) & np.uint64(2**12))==0)
    allmask = ((df['MASKBITS'] & 2**6) == 0) & ((df['MASKBITS'] & 2**5) == 0) & ((df['MASKBITS'] & 2**7) == 0)
    nobs = (df['NOBS_G'] > 0) & (df['NOBS_R'] > 0) & (df['NOBS_Z'] > 0)
        
    LX = addBackLG(df=df, randoms=randoms, survey=survey)
    
    GeoCut = {'BS':BS | LX,
              'MS':MS | LX,
              'GC':GC | LX,
              'LG':LG | LX,
              'allmask':allmask | LX,
              'nobs':nobs | LX
             }
    
    return GeoCut

#
def getPhotCuts(df, mycat=False, survey='main'):
    
    cols = df.dtype.names
    
    LX = addBackLG(df=df, randoms=False, survey=survey)
    
    if 'TYPE' in cols: psftype = df['TYPE']
    elif 'MORPHTYPE' in cols: psftype = df['MORPHTYPE']
        
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') | (psftype == 'PSF ') | (psftype == b'PSF '))
    
    if mycat:
        rmag = df['RMAG']
        gmag = df['GMAG']
        zmag = df['ZMAG']
        w1mag = df['W1MAG']
        rfibmag = df['RFIBERMAG']
        G = df['G']
    else:
        rmag = flux_to_mag((df['FLUX_R']/df['MW_TRANSMISSION_R']).clip(1e-16))
        gmag = flux_to_mag((df['FLUX_G']/df['MW_TRANSMISSION_G']).clip(1e-16))
        zmag = flux_to_mag((df['FLUX_Z']/df['MW_TRANSMISSION_Z']).clip(1e-16))
        w1mag = flux_to_mag((df['FLUX_W1']/df['MW_TRANSMISSION_W1']).clip(1e-16))
        rfibmag = flux_to_mag((df['FIBERFLUX_R']/df['MW_TRANSMISSION_R']).clip(1e-16))
        G = df['GAIA_PHOT_G_MEAN_MAG']
        
    keys = ['MASKBITS', 'FLUX_R', 'FLUX_G', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FIBERFLUX_R',
            'MW_TRANSMISSION_R', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
            'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FIBERTOTFLUX_R',
            'TYPE', 'NOBS_R', 'NOBS_G', 'NOBS_Z'
           ]
    wisepar = {}
    for key in keys:
        wisepar[key] = df[key]
    wisepar['G'] = G
    agns = get_bgsw2(cat=wisepar)
    nomask = np.zeros_like(df['RA'], dtype='?')

    #Photometric cuts CATALOGUE
    STARS = get_stars(gaiagmag=G, fluxr=df['FLUX_R'])
    GAL = get_galaxies(gaiagmag=G, fluxr=df['FLUX_R'])
    
    #BGS WISE
    #bgswise = get_bgswise(gaiagmag=G, fluxr=df['FLUX_R'], zmag=zmag, w1mag=w1mag, gmag=gmag, rmag=rmag, fluxivarw1=df['FLUX_IVAR_W1'], fluxw1=df['FLUX_W1'])
    
    GALSV = get_galaxies_sv(gaiagmag=G, fluxr=df['FLUX_R'], psflike=psflike)
    
    FMC = nomask.copy()
    FMC |= ((rfibmag < (2.9 + 1.2) + rmag) & (rmag < 17.1))
    FMC |= ((rfibmag < 21.2) & (rmag < 18.3) & (rmag > 17.1))
    FMC |= ((rfibmag < 2.9 + rmag) & (rmag > 18.3))
    
    FMC2 = nomask.copy()
    delta = 1.0
    FMC2 |= ((rfibmag < (2.9 + 1.2 + delta) + rmag) & (rmag < 18.8 - delta))
    FMC2 |= ((rfibmag < 22.9) & (rmag < 20) & (rmag > 18.8 - delta))
    FMC2 |= ((rfibmag < 2.9 + rmag) & (rmag > 20))
    
    CC = ~nomask.copy()
    CC &= ((gmag - rmag) > -1.)
    CC &= ((gmag - rmag) < 4.)
    CC &= ((rmag - zmag) > -1.)
    CC &= ((rmag - zmag) < 4.)

    QC_FM = ~nomask.copy()
    QC_FM &= (df['FRACMASKED_R'] < 0.4)
    QC_FM &= (df['FRACMASKED_G'] < 0.4)
    QC_FM &= (df['FRACMASKED_Z'] < 0.4)
    
    QC_FI = ~nomask.copy()
    QC_FI &= (df['FRACIN_R'] > 0.3) 
    QC_FI &= (df['FRACIN_G'] > 0.3) 
    QC_FI &= (df['FRACIN_Z'] > 0.3) 
    
    QC_FF = ~nomask.copy()
    QC_FF &= (df['FRACFLUX_R'] < 5.) 
    QC_FF &= (df['FRACFLUX_G'] < 5.)  
    QC_FF &= (df['FRACFLUX_Z'] < 5.) 
    
    QC_IVAR = ~nomask.copy()
    QC_IVAR &= (df['FLUX_IVAR_R'] > 0.)  
    QC_IVAR &= (df['FLUX_IVAR_G'] > 0.)  
    QC_IVAR &= (df['FLUX_IVAR_Z'] > 0.) 
    
    #
    QC_FM2 = nomask.copy()
    QC_FM2 |= ((df['FRACMASKED_R'] < 0.4) & (df['FRACMASKED_G'] < 0.4))
    QC_FM2 |= ((df['FRACMASKED_R'] < 0.4) & (df['FRACMASKED_Z'] < 0.4))
    QC_FM2 |= ((df['FRACMASKED_G'] < 0.4) & (df['FRACMASKED_Z'] < 0.4))
    
    QC_FI2 = nomask.copy()
    QC_FI2 |= ((df['FRACIN_R'] > 0.3) & (df['FRACIN_G'] > 0.3))
    QC_FI2 |= ((df['FRACIN_R'] > 0.3) & (df['FRACIN_Z'] > 0.3))
    QC_FI2 |= ((df['FRACIN_G'] > 0.3) & (df['FRACIN_Z'] > 0.3))
    
    QC_FF2 = nomask.copy()
    QC_FF2 |= ((df['FRACFLUX_R'] < 5.) & (df['FRACFLUX_G'] < 5.))
    QC_FF2 |= ((df['FRACFLUX_R'] < 5.) & (df['FRACFLUX_Z'] < 5.))
    QC_FF2 |= ((df['FRACFLUX_G'] < 5.) & (df['FRACFLUX_Z'] < 5.))
    
    rfibcol = nomask.copy()
    schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - rmag) + 1.2
    rfibcol |= ((rfibmag < 20.75) | ((rfibmag < 21.5) & (schlegel_color > 0.)))
    

    
    PhotCut = {'SG':GAL | LX,
               'SGSV':GALSV | LX,
              'FMC':FMC | LX,
              'FMC2':FMC2 | LX,
              'CC':CC | LX,
              'QC_FM':QC_FM | LX,
              'QC_FI':QC_FI | LX,
              'QC_FF':QC_FF | LX,
              'QC_FM2':QC_FM2 | LX,
              'QC_FI2':QC_FI2 | LX,
              'QC_FF2':QC_FF2 | LX,
              'QC_IVAR':QC_IVAR | LX,
              'rfibcol':rfibcol,
               'agns':agns
             }
    
    return PhotCut

def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def get_bgs(df, mycat=False):
    
    if mycat:
        rmag = df['RMAG']
        rfibmag = df['RFIBERMAG']
    else:
        rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
        rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
    
    geocuts = getGeoCuts(df)
    photcuts = getPhotCuts(df, mycat=mycat)
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgslist = ['BS', 'LG', 'GC', 'nobs', 'SG', 'FMC2', 'CC', 'QC_FM', 'QC_FI', 'QC_FF']
    
    bgs = np.ones_like(df['RA'], dtype='?')
    bgs_bright = bgs.copy()
    bgs_faint = bgs.copy()
    bgs_any = bgs.copy()
    
    for key in bgslist:
        bgs &= bgscuts[key]
        
    bgs_bright = (bgs) & (rmag < 19.5)
    bgs_faint = (bgs) & (rmag > 19.5) & (rmag < 20.0)
    bgs_any = (bgs_bright) | (bgs_faint)
    #for key, val in zip(bgscuts.keys(), bgscuts.values()):
    #    if (key == 'allmask') or (key == 'MS') or (key == 'MS') or (key == 'MS'): continue
    #    else: bgs &= val
        
    return bgs_any, bgs_bright, bgs_faint

def get_bgs_sv3(df, mycat=False, south=True):
    
    if mycat:
        rmag = df['RMAG']
        gmag = df['GMAG']
        zmag = df['ZMAG']
        w1mag = df['W1MAG']
        rfibmag = df['RFIBERMAG']
    else:
        rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
        gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
        zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
        w1mag = flux_to_mag(df['FLUX_W1']/df['MW_TRANSMISSION_W1'])
        rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
        
    fibertot_r = df["FIBERTOTFLUX_R"]#/df["MW_TRANSMISSION_R"]
    fibertomag_r = flux_to_mag(fibertot_r)
    
    geocuts = getGeoCuts(df)
    photcuts = getPhotCuts(df, mycat=mycat)
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgslist = ['BS', 'GC', 'nobs', 'SG', 'FMC2', 'CC']
    
    bgs = np.ones_like(df['RA'], dtype='?')
    bgs_bright = bgs.copy()
    bgs_faint = bgs.copy()
    bgs_wise = bgs.copy()
    bgs_any = bgs.copy()
    rfibcol = ~bgs.copy()
    
    #BASS r-mag offset with DECaLS
    offset = 0.04
    
    for key in bgslist:
        bgs &= bgscuts[key]
        
    if south:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - rmag) + 1.2
        rfibcol |= ((rfibmag < 20.75) | ((rfibmag < 21.5) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < 19.5) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= 19.5) & (rmag < 20.3) & (rfibcol)
    else:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - (rmag-offset)) + 1.2
        rfibcol |= ((rfibmag < (20.75+offset)) | ((rfibmag < (21.5+offset)) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < (19.5+offset)) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= (19.5+offset)) & (rmag < (20.3)) & (rfibcol)
    
    bgs_wise = bgscuts['agns']
    bgs_any = (bgs_bright) | (bgs_faint) | (bgs_wise)
    
    #for key, val in zip(bgscuts.keys(), bgscuts.values()):
    #    if (key == 'allmask') or (key == 'MS') or (key == 'MS') or (key == 'MS'): continue
    #    else: bgs &= val
        
    return bgs_any, bgs_bright, bgs_faint, bgs_wise

def get_bgs_sv3_corrected(df, mycat=False, south=True):
    
    if mycat:
        rmag = df['RMAG']
        gmag = df['GMAG']
        zmag = df['ZMAG']
        w1mag = df['W1MAG']
        rfibmag = df['RFIBERMAG']
    else:
        rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
        gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
        zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
        w1mag = flux_to_mag(df['FLUX_W1']/df['MW_TRANSMISSION_W1'])
        rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
        
    fibertot_r = df["FIBERTOTFLUX_R"]#/df["MW_TRANSMISSION_R"]
    fibertomag_r = flux_to_mag(fibertot_r)
    
    geocuts = getGeoCuts(df)
    photcuts = getPhotCuts(df, mycat=mycat)
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgslist = ['BS', 'GC', 'nobs', 'SG', 'FMC2', 'CC']
    
    bgs = np.ones_like(df['RA'], dtype='?')
    bgs_bright = bgs.copy()
    bgs_faint = bgs.copy()
    bgs_wise = bgs.copy()
    bgs_any = bgs.copy()
    rfibcol = ~bgs.copy()
    
    #BASS r-mag offset with DECaLS
    offset = 0.04
    
    for key in bgslist:
        bgs &= bgscuts[key]
        
    if south:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - rmag) + 1.2
        rfibcol |= ((rfibmag < 20.75) | ((rfibmag < 21.5) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < 19.5) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= 19.5) & (rmag < 20.3) & (rfibcol)
    else:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - (rmag-offset)) + 1.2
        rfibcol |= ((rfibmag < (20.75+offset)) | ((rfibmag < (21.5+offset)) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < (19.5+offset)) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= (19.5+offset)) & (rmag < (20.3+offset)) & (rfibcol)
    
    bgs_wise = bgscuts['agns']
    bgs_any = (bgs_bright) | (bgs_faint) | (bgs_wise)
    
    #for key, val in zip(bgscuts.keys(), bgscuts.values()):
    #    if (key == 'allmask') or (key == 'MS') or (key == 'MS') or (key == 'MS'): continue
    #    else: bgs &= val
        
    return bgs_any, bgs_bright, bgs_faint, bgs_wise

def get_bgs_1percent(df, mycat=False, south=True):
    
    if mycat:
        rmag = df['RMAG']
        gmag = df['GMAG']
        zmag = df['ZMAG']
        w1mag = df['W1MAG']
        rfibmag = df['RFIBERMAG']
    else:
        rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
        gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
        zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
        w1mag = flux_to_mag(df['FLUX_W1']/df['MW_TRANSMISSION_W1'])
        rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
        
    fibertot_r = df["FIBERTOTFLUX_R"]#/df["MW_TRANSMISSION_R"]
    fibertomag_r = flux_to_mag(fibertot_r)
    
    geocuts = getGeoCuts(df)
    photcuts = getPhotCuts(df, mycat=mycat)
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgslist = ['BS', 'GC', 'nobs', 'SG', 'FMC2', 'CC']
    
    bgs = np.ones_like(df['RA'], dtype='?')
    bgs_bright = bgs.copy()
    bgs_faint = bgs.copy()
    bgs_wise = bgs.copy()
    bgs_any = bgs.copy()
    rfibcol = ~bgs.copy()
    
    #BASS r-mag offset with DECaLS
    offset = 0.04
    
    for key in bgslist:
        bgs &= bgscuts[key]
        
    if south:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - rmag) + 1.2
        rfibcol |= ((rfibmag < 20.75) | ((rfibmag < 21.5) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < 19.5) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= 19.5) & (rmag < 20.175) & (rfibcol)
    else:
        
        schlegel_color = (zmag - w1mag) - 3/2.5 * (gmag - (rmag-offset)) + 1.2
        rfibcol |= ((rfibmag < (20.75+offset)) | ((rfibmag < (21.5+offset)) & (schlegel_color > 0.)))
        
        bgs_bright = (bgs) & (rmag < (19.5+offset)) & (rmag > 12.0) & (fibertomag_r > 15)
        bgs_faint = (bgs) & (rmag >= (19.5+offset)) & (rmag < (20.220)) & (rfibcol)
    
    bgs_wise = bgscuts['agns']
    bgs_any = (bgs_bright) | (bgs_faint) | (bgs_wise)
    
    #for key, val in zip(bgscuts.keys(), bgscuts.values()):
    #    if (key == 'allmask') or (key == 'MS') or (key == 'MS') or (key == 'MS'): continue
    #    else: bgs &= val
        
    return bgs_any, bgs_bright, bgs_faint, bgs_wise

def get_bgs_sv(df, mycat=False):
    
    if mycat:
        rmag = df['RMAG']
        rfibmag = df['RFIBERMAG']
    else:
        rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
        rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
    
    geocuts = getGeoCuts(df, survey='sv')
    photcuts = getPhotCuts(df, mycat=mycat, survey='sv')
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgslist = ['BS', 'GC', 'nobs', 'SGSV', 'CC']
    #bgslist = ['BS', 'GC', 'nobs', 'SGSV', 'CC', 'QC_FM', 'QC_FI', 'QC_FF']
    bgslist_WQC = ['BS', 'GC', 'nobs', 'SGSV', 'CC', 'QC_FM', 'QC_FI', 'QC_FF']
    #lowlist = ['BS', 'GC', 'nobs', 'SGSV', 'CC', 'QC_FM', 'QC_FI', 'QC_FF']
    bgs_sv = np.ones_like(df['RA'], dtype='?')
    bgs_sv_bright = bgs_sv.copy()
    bgs_sv_faint = bgs_sv.copy()
    bgs_sv_faint_ext = bgs_sv.copy()
    bgs_sv_fibmag = bgs_sv.copy()
    bgs_sv_lowq = ~bgs_sv.copy()
    #bgs_sv_lowq_WQC = ~bgs_sv.copy()
    bgs_sv_any = bgs_sv.copy()
    #lowq = ~bgs_sv.copy()
    
    for key in bgslist:
        bgs_sv &= bgscuts[key]
        #if key != 'SGSV': 
        #    bgs_sv_lowq |= ((bgscuts['SGSV']) & (bgscuts['BS']) & (bgscuts['GC']) & (~bgscuts[key]))
    
    # This add the QCs to lowq sample
    for key in bgslist_WQC:
        if key != 'SGSV': 
            bgs_sv_lowq |= ((bgscuts['SGSV']) & (bgscuts['BS']) & (bgscuts['GC']) & (~bgscuts[key]))
        
    bgs_sv_bright = (bgs_sv) & (rmag < 19.5)
    bgs_sv_faint = (bgs_sv) & (rmag >= 19.5) & (rmag < 20.1)
    bgs_sv_faint_ext = (bgs_sv) & (rmag >= 20.1) & (rmag < 20.5) & (rfibmag > 21.0511)
    bgs_sv_fibmag = (bgs_sv) & (rmag >= 20.1) & (rmag < 20.5) & (rfibmag < 21.0511)
    bgs_sv_lowq &= (rmag < 20.1)
    #bgs_sv_lowq_WQC &= (rmag < 20.1)
    bgs_sv_any = (bgs_sv_bright) | (bgs_sv_faint) | (bgs_sv_faint_ext) | (bgs_sv_fibmag) | (bgs_sv_lowq)
    #bgs_sv_any_WQC = (bgs_sv_bright) | (bgs_sv_faint) | (bgs_sv_faint_ext) | (bgs_sv_fibmag) | (bgs_sv_lowq_WQC)
        
    return bgs_sv_any, bgs_sv_bright, bgs_sv_faint, bgs_sv_faint_ext, bgs_sv_fibmag, bgs_sv_lowq #, bgs_sv_any_WQC, bgs_sv_lowq_WQC

def get_bgsw2(cat):    
    
    fluxr = (cat["FLUX_R"]/cat["MW_TRANSMISSION_R"])
    fluxg = (cat["FLUX_G"]/cat["MW_TRANSMISSION_G"])
    fluxz = (cat["FLUX_Z"]/cat["MW_TRANSMISSION_Z"])
    fluxw1 = (cat["FLUX_W1"]/cat["MW_TRANSMISSION_W1"])
    fluxw2 = (cat["FLUX_W2"]/cat["MW_TRANSMISSION_W2"])
    
    rmag = flux_to_mag(fluxr.clip(1e-16))
    gmag = flux_to_mag(fluxg.clip(1e-16))
    zmag = flux_to_mag(fluxz.clip(1e-16))
    w1mag = flux_to_mag(fluxw1.clip(1e-16))
    w2mag = flux_to_mag(fluxw2.clip(1e-16))
    rfibmag = flux_to_mag((cat['FIBERFLUX_R']/cat['MW_TRANSMISSION_R']).clip(1e-16))

    error = (cat["FLUX_IVAR_G"]>0) & (cat["FLUX_IVAR_R"]>0) & (cat["FLUX_IVAR_Z"]>0)
    Grr = cat['G'] - 22.5 + 2.5*np.log10(cat['FLUX_R'])
    fibertot_r = cat["FIBERTOTFLUX_R"]/cat["MW_TRANSMISSION_R"]
    fibertomag_r = flux_to_mag(fibertot_r)
    w1sn = cat['FLUX_W1'] * np.sqrt(cat['FLUX_IVAR_W1'])
    w2sn = cat['FLUX_W2'] * np.sqrt(cat['FLUX_IVAR_W2'])
    psftype = cat['TYPE']
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') | (psftype == 'PSF ') | (psftype == b'PSF '))
    psfsel = (((cat['TYPE'] == 'PSF') & (rmag < 17.5)) | (cat['TYPE'] != 'PSF'))
    nobs = (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)
    BS = (np.uint64(cat['MASKBITS']) & np.uint64(2**1))==0
    GC = (np.uint64(cat['MASKBITS']) & np.uint64(2**13))==0

    agnw2 = (zmag - w2mag - (gmag - rmag) > -0.5) &\
            (w2sn > 10) &\
            ((cat["MASKBITS"]&2**(9))==0) &\
            (fluxw2>0)
    
    quality = (fluxr>0) & (fluxg>0) & (fluxz>0) &\
          (error) & (nobs) & (BS) & (GC) &\
            ((cat['G'] == 0) | ((cat['G'] != 0) & (cat['G'] > 16))) &\
            (rmag < 20.3) & (rmag > 16) &\
            (rfibmag < 22) &\
            (((rfibmag  < 21.5) & (rmag > 19.5)) | (rmag < 19.5)) &\
            (psfsel) &\
            (fibertomag_r > 15)
    
    stars = ((cat['G'] != 0) & (Grr < 0.6)) | ((cat['G'] == 0) & (psflike))

    additional = ((w1mag - w2mag) > -0.2) &\
                  (zmag - w1mag - (gmag - rmag) > -0.7) &\
                  (w1sn > 10)

    agn_ext = (agnw2) & (additional)
    
    AGN = (agn_ext) & (quality) & ~((stars) & (~agn_ext)) & (Grr < 0.6) & (cat['G'] != 0)
    
    return AGN

def get_stars(gaiagmag, fluxr):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    GAIA_STAR = np.ones_like(gaiagmag, dtype='?')
    GAIA_STAR &= (Grr  <  0.6) & (gaiagmag != 0)
    
    return GAIA_STAR

def get_galaxies(gaiagmag, fluxr):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    GAIA_GAL = np.ones_like(gaiagmag, dtype='?')
    GAIA_GAL &= (Grr  >  0.6) | (gaiagmag == 0)
    
    return GAIA_GAL

def get_galaxies_sv(gaiagmag, fluxr, psflike):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    GAIA_GAL = np.ones_like(gaiagmag, dtype='?')
    GAIA_GAL &= ((Grr  >  0.6) | (gaiagmag == 0)) | ((Grr < 0.6) & (~psflike) & (gaiagmag != 0))
    
    return GAIA_GAL

def bgsbut(bgsbits=None, rmag=None, pop=None, bgsmask=None, rlimit=20, bgslist=None):
    
    if bgslist is None:
        bgslist = ['BS', 'LG', 'GC', 'nobs', 'SG', 'FMC2', 'CC', 'QC_FM', 'QC_FI', 'QC_FF']
    if pop is not None:
        [bgslist.remove(i) for i in pop]
    
    bgsbut = np.ones_like(rmag, dtype=bool)
    for key in bgslist:
        keep = ((bgsbits & 2**(bgsmask[key])) != 0)
        bgsbut &= keep
    bgsbut &= rmag < rlimit
    
    return bgsbut

def get_bgswise(gaiagmag, fluxr, zmag, w1mag, gmag, rmag, fluxivarw1, fluxw1):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    w1sn = fluxw1 * np.sqrt(fluxivarw1)
    bgswise = np.ones_like(gaiagmag, dtype='?')
    bgswise &= (Grr < 0.6) & (Grr > -1) & (zmag - w1mag - (gmag - rmag) > -0.5) & (w1sn > 10)
    
    return bgswise
