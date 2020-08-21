#
import numpy as np
import sys, os, time, argparse, glob
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
from astropy.coordinates import SkyCoord
from matplotlib.ticker import NullFormatter
import astropy.units as units
from astropy import units as u
import pandas as pd
from astropy.io import ascii
#from photometric_def import get_stars, get_galaxies, masking, results
from scipy import optimize
import pygraphviz as pgv
from PIL import Image
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from matplotlib_venn import venn3, venn3_circles
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

#sys.path.insert(0, '/global/homes/q/qmxp55/DESI/bgstargets/py')


#import raichoorlib
#np.seterr(divide='ignore') # ignode divide by zero warnings
import astropy.io.fits as fits
import healpy as hp

def getStats(cat=None, hpdict=None, bgsmask=None, rancuts=None, CurrentMask=None, PrevMask=None, 
                 reg='decals', regcat=None, regran=None, program='main'):
    
    #from astropy.table import Table
    Tab = []
    GMT = np.zeros_like(regcat, dtype='?')
    GMT_ran = np.zeros_like(regran, dtype='?')
    PMT = GMT.copy()
    PMT_ran = GMT_ran.copy()
        
    #if (regcat is not None) & (regran is not None): print('region set...')
    #elif (regcat is None) & (regran is None): regcat, regran = ~GMT.copy(), ~GMT_ran.copy()
    #else: raise ValueError('regcat and regran both have to be None-type or non None-type.')
     
    # area in region
    if reg == 'desi': Areg = hpdict['area_all']
    else: Areg = hpdict['bgsarea_'+reg]
    # Number of randoms in region
    NRreg = np.sum(regran)
    #
    #for BGS main
    B, F = cat['RMAG'] < 19.5, np.logical_and(cat['RMAG'] < 20, cat['RMAG'] > 19.5)
    #for BGS SV
    if program == 'sv':
        F_SV = np.logical_and(cat['RMAG'] < 20.1, cat['RMAG'] > 19.5)
        FE_SV = (cat['RMAG'] > 20.1) & (cat['RMAG'] < 20.5) & (cat['RFIBERMAG'] > 21.0511)
        FIBM_SV = (cat['RMAG'] > 20.1) & (cat['RFIBERMAG'] < 21.0511)
        
    if PrevMask is not None:
        PM_lab = '|'.join(PrevMask)
        for i in PrevMask:
            PMT |= (cat['BGSBITS'] & 2**(bgsmask[i])) == 0
            if i in rancuts.keys(): PMT_ran |= ~rancuts[i]   
    else:
        PM_lab = 'None'
    
    for i in CurrentMask:
        
        if i in rancuts.keys(): A_i = (np.sum((~rancuts[i] & (~PMT_ran) & (regran)))/NRreg)*(Areg)
        else: A_i = 0.
        bgscut = (cat['BGSBITS'] & 2**(bgsmask[i])) == 0
        #eta_B_i_in = np.sum((GeoCutsDict[i]) & (B) & (~PMT))/(A_i) #density over the geometric area
        #eta_F_i_in = np.sum((GeoCutsDict[i]) & (F) & (~PMT))/(A_i) #density over the geometric area
        if program == 'main':
            eta_B_i = np.sum((bgscut) & (B) & (~PMT) & (regcat))/(Areg) #density over the total area
            eta_F_i = np.sum((bgscut) & (F) & (~PMT) & (regcat))/(Areg) #density over the total area
            Tab.append([i, round(A_i*(100/Areg), 2), round(eta_B_i,2), round(eta_F_i,2)])
        elif program == 'sv':
            eta_B_i = np.sum((bgscut) & (B) & (~PMT) & (regcat))/(Areg)
            eta_F_SV_i = np.sum((bgscut) & (F_SV) & (~PMT) & (regcat))/(Areg)
            eta_FE_SV_i = np.sum((bgscut) & (FE_SV) & (~PMT) & (regcat))/(Areg)
            eta_FIBM_SV_i = np.sum((bgscut) & (FIBM_SV) & (~PMT) & (regcat))/(Areg)
            Tab.append([i, round(A_i*(100/Areg), 2), round(eta_B_i,2), round(eta_F_SV_i,2), round(eta_FE_SV_i,2), round(eta_FIBM_SV_i,2)])
            
        GMT |= bgscut
        if i in rancuts.keys(): GMT_ran |= ~rancuts[i]  
    
    lab = '|'.join(CurrentMask)
    lab_in = '(%s)' %(lab)
    lab_out = '~(%s)*' %(lab)
    lab_out2 = '~(%s)' %(lab)
    
    A_GMT_in = (np.sum((GMT_ran) & (~PMT_ran) & (regran))/NRreg)*(Areg)
    if program == 'main':
        eta_B_GMT_in_1 = np.sum((GMT) & (B) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_F_GMT_in_1 = np.sum((GMT) & (F) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
    elif program == 'sv':
        eta_B_GMT_in_1 = np.sum((GMT) & (B) & (~PMT) & (regcat))/(Areg)
        eta_F_SV_GMT_in_1 = np.sum((GMT) & (F_SV) & (~PMT) & (regcat))/(Areg)
        eta_FE_SV_GMT_in_1 = np.sum((GMT) & (FE_SV) & (~PMT) & (regcat))/(Areg)
        eta_FIBM_SV_GMT_in_1 = np.sum((GMT) & (FIBM_SV) & (~PMT) & (regcat))/(Areg)
        

    A_GMT_out = (np.sum((~GMT_ran) & (~PMT_ran) & (regran))/NRreg)*(Areg)
    if program == 'main':
        eta_B_GMT_out_1 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_B_GMT_out_2 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
        eta_F_GMT_out_1 = np.sum((~GMT) & (F) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_F_GMT_out_2 = np.sum((~GMT) & (F) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
    elif program == 'sv':
        eta_B_GMT_out_1 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_B_GMT_out_2 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
        
        eta_F_SV_GMT_out_1 = np.sum((~GMT) & (F_SV) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_F_SV_GMT_out_2 = np.sum((~GMT) & (F_SV) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
        
        eta_FE_SV_GMT_out_1 = np.sum((~GMT) & (FE_SV) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_FE_SV_GMT_out_2 = np.sum((~GMT) & (FE_SV) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
        
        eta_FIBM_SV_GMT_out_1 = np.sum((~GMT) & (FIBM_SV) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
        eta_FIBM_SV_GMT_out_2 = np.sum((~GMT) & (FIBM_SV) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
    
    
    if program == 'main':
        if len(CurrentMask) > 1:
            Tab.append([lab_in, round(A_GMT_in*(100/Areg),2), round(eta_B_GMT_in_1,2), round(eta_F_GMT_in_1,2)])
        Tab.append([lab_out, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_1,2), round(eta_F_GMT_out_1,2)])
        Tab.append([lab_out2, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_2,2), round(eta_F_GMT_out_2,2)])
    
        Tab = np.transpose(Tab)
        t = Table([Tab[0], Tab[1], Tab[2], Tab[3]], 
              names=('GM','$f_{A}$ [$\%$]', '$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]'),
                    dtype=('S', 'f8', 'f8', 'f8'))
    elif program == 'sv':
        if len(CurrentMask) > 1:
            Tab.append([lab_in, round(A_GMT_in*(100/Areg),2), round(eta_B_GMT_in_1,2), round(eta_F_SV_GMT_in_1,2), round(eta_FE_SV_GMT_in_1,2), round(eta_FIBM_SV_GMT_in_1,2)])
        Tab.append([lab_out, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_1,2), round(eta_F_SV_GMT_out_1,2), round(eta_FE_SV_GMT_out_1,2), round(eta_FIBM_SV_GMT_out_1,2)])
        Tab.append([lab_out2, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_2,2), round(eta_F_SV_GMT_out_2,2), round(eta_FE_SV_GMT_out_2,2), round(eta_FIBM_SV_GMT_out_2,2)])
    
        Tab = np.transpose(Tab)
        t = Table([Tab[0], Tab[1], Tab[2], Tab[3], Tab[4], Tab[5]], 
              names=('GM','$f_{A}$ [$\%$]', '$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]', '$\eta_{FE}$ [deg$^2$]', '$\eta_{FIBM}$ [deg$^2$]'),
                    dtype=('S', 'f8', 'f8', 'f8', 'f8', 'f8'))
    
    print('Previous Cuts: (%s)' %(PM_lab))
    print('Current Cuts: %s' %(lab_in))
                                    
    return t


def flow(cat=None, hpdict=None, bgsmask=None, rancuts=None, order=None, reg=None, 
             regcat=None, regran=None, file=None, dr=None, program='main'):
    
    # add GRAPHVIZ bin files to PATH, otherwise it doesn't find them
    os.environ['PATH'] = '/global/u2/q/qmxp55/bin'
    
    if order is None:
        raise ValueError('define the order of flow chart.')
    
    T = Table()
    if reg == 'desi': Areg = hpdict['area_all']
    else: Areg = hpdict['bgsarea_'+reg]
    
    #for BGS main
    B, F = cat['RMAG'] < 19.5, np.logical_and(cat['RMAG'] < 20, cat['RMAG'] > 19.5)
    #for BGS SV
    if program == 'sv':
        F_SV = np.logical_and(cat['RMAG'] < 20.1, cat['RMAG'] > 19.5)
        FE_SV = (cat['RMAG'] > 20.1) & (cat['RMAG'] < 20.5) & (cat['RFIBERMAG'] > 21.0511)
        FIBM_SV = (cat['RMAG'] > 20.1) & (cat['RFIBERMAG'] < 21.0511)
    
    #initial density
    if program == 'main':
        den0B = np.sum((regcat) & (B))/Areg
        den0F = np.sum((regcat) & (F))/Areg
    elif program == 'sv':
        den0B = np.sum((regcat) & (B))/Areg
        den0F_SV = np.sum((regcat) & (F_SV))/Areg
        den0FE_SV = np.sum((regcat) & (FE_SV))/Areg
        den0FIBM_SV = np.sum((regcat) & (FIBM_SV))/Areg
    
    #T['SU'] = masking(title='START', submasks=None, details=None)
    #T['SG'] = masking(title='GEOMETRICAL', submasks=None, details=None)
    #T['I'] = masking(title='LS %s (%s)' %(dr, reg.upper()), submasks=['rmag < %2.2g' %(20)], details=None)
    T['I'] = masking(title='LS %s (%s)' %(dr, reg.upper()), submasks=None, details=None)
    if program == 'main':
        T['RI'] = results(a=Areg, b=den0B, f=den0F, stage='ini', per=False)
    elif program == 'sv':
        T['RI'] = results_SV(a=Areg, b=den0B, f=den0F_SV, fe=den0FE_SV, fibm=den0FIBM_SV, stage='ini', per=False)
    
    G=pgv.AGraph(strict=False,directed=True)

    elist = []
    rejLab = []
    #define initial params in flow chart
    #ini = ['SU', 'I', 'RI', 'SG']
    ini = ['I', 'RI']
    for i in range(len(ini) - 1):
        elist.append((list(T[ini[i]]),list(T[ini[i+1]])))
        
    #G.add_edges_from(elist)
    #stages=['SU', 'SG']
    #G.add_nodes_from([list(T[i]) for i in stages], color='green', style='filled')
    nlist=['RI']
    G.add_nodes_from([list(T[i]) for i in nlist], color='lightskyblue', shape='box', style='filled')
    maskings=['I']
    G.add_nodes_from([list(T[i]) for i in maskings], color='lawngreen', style='filled')
        
    #
    for num, sel in enumerate(order):
        
        T['I'+str(num)] = masking(title=' & '.join(sel), submasks=None, details=None)
        
        if num == 0: elist.append((list(T['I']),list(T['I'+str(num)])))
        else: elist.append((list(T['R'+str(num-1)]),list(T['I'+str(num)])))
            
        if num == 0: pm = None
        elif num == 1: pm = order[0]
        else: pm += order[num-1]
            
        if len(sel) > 1: IGMLab_2 = ' | '.join(sel)
        else: IGMLab_2 = sel[0]
        
        t = getStats(cat=cat, hpdict=hpdict, bgsmask=bgsmask, rancuts=rancuts, CurrentMask=sel, PrevMask=pm, 
                 reg=reg, regcat=regcat, regran=regran, program=program)
        
        if program == 'main':
            T['R'+str(num)] = results(a=t[-2][1], b=t[-2][2], f=t[-2][3], b2=t[-1][2], f2=t[-1][3], stage='geo', per=True)
            T['REJ'+str(num)] = results(a=t[-3][1], b=t[-3][2], f=t[-3][3], stage='ini', per=True, title='(%s)' %(IGMLab_2))
        elif program == 'sv':
            T['R'+str(num)] = results_SV(a=t[-2][1], b=t[-2][2], f=t[-2][3], fe=t[-2][4], fibm=t[-2][5], b2=t[-1][2], f2=t[-1][3], fe2=t[-1][4], fibm2=t[-1][5], stage='geo', per=True)
            T['REJ'+str(num)] = results_SV(a=t[-3][1], b=t[-3][2], f=t[-3][3], fe=t[-3][4], fibm=t[-3][5], stage='ini', per=True, title='(%s)' %(IGMLab_2))
            
        elist.append((list(T['I'+str(num)]),list(T['REJ'+str(num)])))
        elist.append((list(T['I'+str(num)]),list(T['R'+str(num)])))
        
        if False in [i in rancuts.keys() for i in sel]: icolor = 'plum'
        else: icolor = 'lightgray'
        
        Rlist=['R'+str(num)]
        G.add_nodes_from([list(T[i]) for i in Rlist], color='lightskyblue', shape='box', style='filled')
        REJlist=['REJ'+str(num)]
        G.add_nodes_from([list(T[i]) for i in REJlist], color='lightcoral', shape='box', style='filled')
        Ilist=['I'+str(num)]
        G.add_nodes_from([list(T[i]) for i in Ilist], color=icolor, style='filled')

        if len(sel) > 1:
            for i, j in enumerate(sel):
                if program == 'main':
                    T['REJ'+str(num)+str(i)] = results(a=t[i][1], b=t[i][2], f=t[i][3], stage='ini', per=True, title=j)
                elif program == 'sv':
                    T['REJ'+str(num)+str(i)] = results_SV(a=t[i][1], b=t[i][2], f=t[i][3], fe=t[i][4], fibm=t[i][5], stage='ini', per=True, title=j)
                    
                elist.append((list(T['REJ'+str(num)]),list(T['REJ'+str(num)+str(i)])))
                
                REJilist=['REJ'+str(num)+str(i)]
                G.add_nodes_from([list(T[i]) for i in REJilist], color='coral', shape='box', style='filled')
        
    #
    if file is None:
        pathdir = os.getcwd()+'/'+'results'+'_'+reg
        if not os.path.isdir(pathdir): os.makedirs(pathdir)
        file = pathdir+'/'+'flow'
        
        
    G.add_edges_from(elist)
    G.write('%s.dot' %(file)) # write to simple.dot
    BB=pgv.AGraph('%s.dot' %(file)) # create a new graph from file
    BB.layout(prog='dot') # layout with default (neato)
    BB.draw('%s.png' %(file)) # draw png
    #os.system('convert ' + file + '.ps ' + file + '.png')
    flow = Image.open('%s.png' %(file))

    return flow, elist, T

def results(a=None, b=None, f=None, b2=None, f2=None, stage='geo', per=True, title=None):
    boldblack = ''#'\033[0;30;1m'
    normal = ''#'\033[0;30;0m'
    gray = ''#'\033[0;30;37m'
    
    if per:
        n = '%'
    else:
        n = 'sq.d'
    
    if stage=='geo':
        R1 = '%s Area: %s%.2f (%s) \n' %(boldblack, normal, a, n)
        R2 = '%s Bright*: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R3 = '%s Faint*: %s%.2f (1/sq.d) \n' %(boldblack, normal, f)
    
        R4 = '%s Bright: %.2f (1/sq.d) \n' %(gray, b2)
        R5 = '%s Faint: %.2f (1/sq.d)' %(gray, f2)
        
        return [R1+R2+R3+R4+R5]
    
    if stage=='photo':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Bright: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s%.2f (1/sq.d)' %(boldblack, normal, f)
        
        return [R]
    
    if stage=='ini':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Area: %s %.2f (%s) \n' %(boldblack, normal, a, n)
        R += '%s Bright: %s %.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s %.2f (1/sq.d)' %(boldblack, normal, f)
        
        return [R]
    
def results_SV(a=None, b=None, f=None, fe=None, fibm=None, b2=None, f2=None, fe2=None, fibm2=None, stage='geo', per=True, title=None):
    boldblack = ''#'\033[0;30;1m'
    normal = ''#'\033[0;30;0m'
    gray = ''#'\033[0;30;37m'
    
    if per:
        n = '%'
    else:
        n = 'sq.d'
    
    if stage=='geo':
        R1 = '%s Area: %s%.2f (%s) \n' %(boldblack, normal, a, n)
        R2 = '%s Bright*: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R3 = '%s Faint*: %s%.2f (1/sq.d) \n' %(boldblack, normal, f)
        R4 = '%s Faint ext*: %s%.2f (1/sq.d) \n' %(boldblack, normal, fe)
        R5 = '%s Fibre mag*: %s%.2f (1/sq.d) \n' %(boldblack, normal, fibm)
    
        R6 = '%s Bright: %.2f (1/sq.d) \n' %(gray, b2)
        R7 = '%s Faint: %.2f (1/sq.d) \n' %(gray, f2)
        R8 = '%s Faint ext: %.2f (1/sq.d) \n' %(gray, fe2)
        R9 = '%s Fibre mag: %.2f (1/sq.d)' %(gray, fibm2)
        
        return [R1+R2+R3+R4+R5+R6+R7+R8+R9]
    
    if stage=='photo':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Bright: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s%.2f (1/sq.d) \n' %(boldblack, normal, f)
        R += '%s Faint ext: %s%.2f (1/sq.d) \n' %(boldblack, normal, fe)
        R += '%s Fibre mag: %s%.2f (1/sq.d)' %(boldblack, normal, fibm)
        
        return [R]
    
    if stage=='ini':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Area: %s %.2f (%s) \n' %(boldblack, normal, a, n)
        R += '%s Bright: %s %.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s %.2f (1/sq.d) \n' %(boldblack, normal, f)
        R += '%s Faint ext: %s %.2f (1/sq.d) \n' %(boldblack, normal, fe)
        R += '%s Fibre mag: %s %.2f (1/sq.d)' %(boldblack, normal, fibm)
        
        return [R]
    
def masking(title, submasks, details):
    
    if details or submasks is not None:
        R = '%s \n\n' %(title)
    else:
        R = '%s' %(title)
    
    if submasks is not None:
        N = len(submasks)
    if details is not None:
        N = len(details)
        
    if details and submasks is not None:
        for i in range(N):
            if i<len(submasks)-1:
                R +='%i) %s \n %s \n' %(i+1, submasks[i], details[i])
            else:
                R +='%i) %s \n %s' %(i+1, submasks[i], details[i])
                
    elif submasks is not None:
        for i in range(N):
            if i<len(submasks)-1:
                R +='%i) %s \n' %(i+1, submasks[i])
            else:
                R +='%i) %s' %(i+1, submasks[i])
                
    elif details is not None:
        for i in range(N):
            if i<len(details)-1:
                R +='%i) %s \n' %(i+1, details[i])
            else:
                R +='%i) %s' %(i+1, details[i])
    
    return [R]

def hexbin(coord, catmask, n, C=None, bins=None, title=None, cmap='viridis', ylab=True, xlab=True, vline=None, 
           hline=None, fig=None, gs=None, xlim=None, ylim=None, vmin=None, vmax=None, mincnt=1, fmcline=False, 
               file=None, gridsize=(60,60), comp=False, fracs=False, area=None, cbar=None, clab=None, 
                   contour1=None, contour2=None, levels1=None, levels2=None, showmedian=False, plothist=False,
                        reduce_C_function=None):
    
    x, y = coord.keys()
    
    #ax = fig.add_subplot(gs[n])
    
    if plothist:
        left, width = 0.1, 0.85
        bottom, height = 0.1, 0.85
        main = [left, bottom, width, height]
        
        #ax = fig.add_subplot(gs[n])
        ax = fig.add_axes(main, gs[n])
    else:
        ax = fig.add_subplot(gs[n])
    
    if title is not None: ax.set_title(r'%s' %(title), size=20)
    if xlim is None: xlim = limits()[x]
    if ylim is None: ylim = limits()[y]
    masklims = (coord[x] > xlim[0]) & (coord[x] < xlim[1]) & (coord[y] > ylim[0]) & (coord[y] < ylim[1])
    
    if catmask is None: keep = masklims
    else: keep = (catmask) & (masklims)
        
    Ntot = np.sum(keep)
        
    if hline is not None:
        maskhigh = (masklims) & (coord[y] > hline) & (catmask)
        masklow = (masklims) & (coord[y] < hline) & (catmask)
        maskgal = (~masklow) & (catmask)
        
    pos = ax.hexbin(coord[x][keep], coord[y][keep], C=C, gridsize=gridsize, cmap=cmap, 
                    vmin=vmin, vmax=vmax, bins=bins, mincnt=mincnt, alpha=0.8, reduce_C_function=reduce_C_function)
    
    dx = np.abs(xlim[1] - xlim[0])/15.
    dy = np.abs(ylim[1] - ylim[0])/15.
    if comp: ax.text(xlim[0]+dx, ylim[1]-dy, r'comp. %2.3g %%' %(100 * np.sum(pos.get_array())/np.sum(keep)), size=15)
    if fracs: 
        ax.text(xlim[1]-5*dx, ylim[1]-dy, r'Ntot. %i' %(Ntot), size=15)
        if area is not None: ax.text(xlim[1]-5*dx, ylim[1]-2*dy, r'$\eta$. %.2f/deg$^2$' %(Ntot/area), size=15)
        ax.text(xlim[0]+dx, ylim[1]-dy, r'f.gal. %.2f %%' %(100 * np.sum(maskhigh)/Ntot), size=15)
        ax.text(xlim[0]+dx, ylim[0]+dy, r'f.stars. %.2f %%' %(100 * np.sum(masklow)/Ntot), size=15)
    if ylab: ax.set_ylabel(r'%s' %(y), size=25)
    if xlab: ax.set_xlabel(r'%s' %(x), size=25)
    if hline is not None: ax.axhline(hline, ls='--', lw=2, c='r')
    if vline is not None: ax.axvline(vline, ls='--', lw=2, c='r')
    if fmcline: 
        #x_N1 = np.linspace(14.5, 17.1, 4)
        #ax.plot(x_N1, 2.9 + 1.2 + x_N1, color='r', ls='--', lw=2)
        #x_N2 = np.linspace(17.1, 18.3, 4)
        #ax.plot(x_N2, x_N2*0.+21.2, color='r', ls='--', lw=2)
        #x_N3 = np.linspace(18.3, 20.1, 4)
        #ax.plot(x_N3, 2.9 + x_N3, color='r', ls='--', lw=2)
        
        delta = 1.0
        
        x1 = np.linspace(14.1, 18.8-delta, 4)
        x_N2 = np.linspace(18.8-delta, 20, 4)
        x_N3 = np.linspace(20, 22, 4)
        plt.plot(x_N2, x_N2*0+22.9, color='r', ls='--', lw=2)
        plt.plot(x_N3, 2.9 + x_N3, color='r', ls='--', lw=2)
        plt.plot(x1, 2.9 + 1.2 + delta + x1, color='r', ls='--', lw=2)
        
        FMC = np.zeros_like(coord[x], dtype='?')
        
        FMC |= ((coord[y] < (2.9 + 1.2 + delta) + coord[x]) & (coord[x] < 18.8 - delta))
        FMC |= ((coord[y] < 22.9) & (coord[x] < 20.0) & (coord[x] > 18.8 - delta))
        FMC |= ((coord[y] < 2.9 + coord[x]) & (coord[x] > 20.0))
        
        maskhigh = (~FMC) & (keep)
        masklow = (FMC) & (keep)
        ax.text(xlim[1]-8*dx, ylim[1]-dy, r'Ntot. %i' %(Ntot), size=15)
        if area is not None: ax.text(xlim[1]-8*dx, ylim[1]-2*dy, r'$\eta$. %.2f/deg$^2$' %(Ntot/area), size=15)
        ax.text(xlim[1]-5*dx, ylim[0]+dy, r'f.kept. %.2f %%' %(100 * np.sum(masklow)/Ntot), size=15)
        ax.text(xlim[0]+dx, ylim[1]-dy, r'f.rej. %.2f %%' %(100 * np.sum(maskhigh)/Ntot), size=15)
        
    #if bins is not None: clab = r'$\log(N)$'
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    if contour1 is not None:
        if levels1 is None: levels1 = (0, 10)
        bgs_den = density_patch(coord=contour1, xlim=xlim, ylim=ylim, plot=False, nmin=0)
        ax.contour(bgs_den.transpose(), levels=levels1, origin='lower', aspect='equal',
              extent=np.array([xlim[0], xlim[1], xlim[0], xlim[1]]), colors='black', linewidths=4, alpha=0.5)
        
    if contour2 is not None:
        if levels2 is None: levels2 = (0, 10)
        bgs_den = density_patch(coord=contour2, xlim=xlim, ylim=ylim, plot=False, nmin=0)
        ax.contour(bgs_den.transpose(), levels=levels2, origin='lower', aspect='equal',
              extent=np.array([xlim[0], xlim[1], xlim[0], xlim[1]]), colors='red', linewidths=4, alpha=0.5)
        
    if showmedian:
        #compute median and percentiles
        binx = np.linspace(xlim[0], xlim[1], 20)
        binw = (binx[1] - binx[0])/2
        binc, median, lower, upper = [],[],[],[]
        
        for num in range(len(binx)-1):
            keepbins = (keep) & (coord[x] > binx[num]) & (coord[x] < binx[num+1])
            
            if np.sum(keepbins) > 0:
                perc = np.percentile(coord[y][keepbins][np.isfinite(coord[y][keepbins])],(3,97))
                binc.append(binx[num] + binw)
                median.append(np.median(coord[y][keepbins]))
                lower.append(perc[0])
                upper.append(perc[1])
            else:
                continue
        
        ax.plot(binc, median, lw=2, c='r')
        ax.fill_between(binc, upper, lower, facecolor='gray', alpha=0.5)
        
    if plothist:
        
        rect_histx = [left, bottom, width, 0.2]
        rect_histy = [left, bottom, 0.2, height]
        axHistx = fig.add_axes(rect_histx)
        axHisty = fig.add_axes(rect_histy)
        binsy = np.linspace(ylim[0], ylim[1], 40)
        binsx = np.linspace(xlim[0], xlim[1], 40)
    
        log = False
        N1 = axHistx.hist(coord[x][keep], bins=binsx, log=log, align='mid', color='r', lw=2, histtype='step')
        N2 = axHisty.hist(coord[y][keep], bins=binsy, log=log, align='mid', color='r', lw=2, histtype='step', orientation='horizontal')
        print('x max:',N1[0].max())
        print('y max:',N2[0].max())
        #axHistx.set_ylim(1, 700)
        #axHisty.set_xlim(1, 700)
        
        axHistx.set_xlim(xlim[0], xlim[1])
        axHisty.set_ylim(ylim[0], ylim[1])
        axHistx.axis('off')
        axHisty.axis('off')
        #ax.axis('off')
        
        #axHistx.yaxis.set_ticks_position('right')
        #axHisty.yaxis.set_ticks_position('right')
        
    
    #clab = r'$N$'
    if cbar in ['horizontal', 'vertical']:
        if plothist:
            if cbar == 'horizontal': cbaxes = fig.add_axes([left, bottom-0.2, width, height*0.08])
            elif cbar == 'vertical': cbaxes = fig.add_axes([0.95, 0.1, 0.06, 0.8])
            cb = fig.colorbar(pos, cax=cbaxes, orientation=cbar)
            
            #cb = fig.colorbar(pos, ax=ax, orientation=cbar, pad=.5)
        else:                                      
            cb = fig.colorbar(pos, ax=ax, orientation=cbar, pad=0.15)
    elif cbar is 'panel':
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        cb = fig.colorbar(pos, cax=cbar_ax)
    else: raise ValueError('cbar is either vertical, horizontal or panel.')
        
    if clab is None: clab = r'$N$'
        
    cb.set_label(label=clab, size=20, weight='bold')
    cb.ax.tick_params(labelsize=16)
    
    if file is not None:
        fig.savefig(file+'.png', bbox_inches = 'tight', pad_inches = 0)
        
    if showmedian:
        return ax, binc, median
    else:
        return ax
        
        

# mollweide plot setting
# http://balbuceosastropy.blogspot.com/2013/09/the-mollweide-projection.html
def set_mwd(ax,org=0):
    # org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis
    ax.set_xlabel('R.A [deg]', size=18)
    ax.xaxis.label.set_fontsize(15)
    ax.set_ylabel('Dec. [deg]', size=18)
    ax.yaxis.label.set_fontsize(15)
    ax.grid(True)
    return True

# convert radec for mollwide
def get_radec_mw(ra,dec,org):
    ra          = np.remainder(ra+360-org,360) # shift ra values
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra          =- ra    # reverse the scale: East to the left
    return np.radians(ra),np.radians(dec)

# plot/xlim settings
def get_systplot(systquant):
    tmparray = np.array([
        'stardens',      [10**(2.4),10**(3.7)],  r'(Stellar density from Gaia/dr2 [deg$^{-2}$])',
        'log10_stardens',      [2.4,3.7],  r'log10(Stellar density from Gaia/dr2 [deg$^{-2}$])',
        'ebv',           [0.01,0.11],'Galactic extinction ebv [mag]',
        'psfsize_g',     [1,2.6],  'g-band psfsize [arcsec]',
        'psfsize_r',     [1,2.6],  'r-band psfsize [arcsec]',
        'psfsize_z',     [1,2.6],  'z-band psfsize [arcsec]',
        'galdepth_g',    [23.3,25.5],'g-band 5sig. galdepth [mag]',
        'galdepth_r',    [23.1,25],'r-band 5sig. galdepth [mag]',
        'galdepth_z',    [21.6,23.9],'z-band 5sig. galdepth [mag]',
        'nobs_g',    [3,4],'g-band NOBS',
        'nobs_r',    [3,4],'r-band NOBS',
        'nobs_z',    [3,4],'z-band NOBS',],
        
        
        dtype='object')
    tmparray = tmparray.reshape(int(tmparray.shape[0]/3),3)
    tmpind   = np.where(tmparray[:,0]==systquant.lower())[0][0]
    return tmparray[tmpind,1], tmparray[tmpind,2]

#
def plot_sysdens(hpdicttmp, namesels, regs, syst, mainreg, xlim=None, n=0, nx=20, clip=False, denslims=False, ylab=True, text=None, weights=False, nside=256, fig=None, gs=None, label=False, ws=None, title=None, onlyweights=False, cols=None, overbyreg=True, percentiles=[1,99], get_values=False):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)

    # plot/xlim settings
    if xlim is None:
        xlim, xlabel = get_systplot(syst)
    else:
        _, xlabel = get_systplot(syst)
        
    #do we have inf or nans within syst boundaries
    tmpsyst0 = hpdicttmp[syst]
    mask = (tmpsyst0>xlim[0]) & (tmpsyst0<xlim[1])
    tmpsyst = tmpsyst0[mainreg & mask]
    
    #print('%i infs and nans found within %s boundaries (%g, %g)' %(np.sum(~np.isfinite(tmpsyst)), syst, xlim[0], xlim[1]))
    #print('%i infs and nans found in target dens. within %s boundaries (%g, %g)' %(np.sum(~np.isfinite(tmpdens)), syst, xlim[0], xlim[1]))
    
    #xlim = tmpsyst[tmpsyst > 0].min(), tmpsyst[tmpsyst > 0].max()
    if clip: xlim = np.percentile(tmpsyst[tmpsyst>0],(1,99))
    xwidth = (xlim[1]-xlim[0])/nx
        
    # initializing plots
    ax = fig.add_subplot(gs[n])
    ## systematics
    ax.plot(xlim,[1.,1.],color='k',linestyle=':')
    ax.set_xlim(xlim)
    ax.set_ylim(0.8,1.2)
    
    delta = (xlim[1] - xlim[0])/15.
    if text is not None: ax.text(xlim[0]+delta, 1.15, text, fontsize=15)
        
    if ylab: ax.set_ylabel(r'$\eta$ / $\overline{\eta}$',fontsize=20)
    ax.set_xlabel(xlabel,fontsize=18)
    ax.grid(True)
    #title = []
    #if clip: title.append('clipped')
    #if denslims: title.append('denslims')
    #ax.set_title(r'%s (%s)' %(namesel, ' & '.join(title)))
    if title is not None: ax.set_title(r'%s' %(title))
        
    ## histogram
    axh = ax.twinx()
    axh.set_xlim(xlim)
    axh.set_ylim(0,8)
    axh.axes.get_yaxis().set_ticks([])
    
    ## systematics
    if cols is None: cols = ['0.5','b','g','r']
    lstys = ['-', '--', '-.']
    #regs = ['all','des','decals','north']
    densmin,densmax = 0,2
    for reg,col1 in zip(regs,cols):
        
        if (reg=='all'):
            isreg    = (mainreg)
            lw,alpha = 3,0.5
        else:
            isreg    = (mainreg) & (hpdicttmp['is'+reg])
            if (reg == regs[0]) & (len(regs) > 1): lw,alpha = 3,0.5
            else: lw,alpha = 1,1.0
                
        for namesel,col2 in zip(namesels, cols):
            
            if len(namesels) > 1: col = col2
            else: col = col1
            
            if (namesel == namesels[0]) & (len(namesels) > 1): lw,alpha = 3,0.5
            else: lw,alpha = 1,1.0
        
            hpdens = (hpdicttmp['south_n'+namesel] + hpdicttmp['north_n'+namesel] ) / (pixarea * hpdicttmp['bgsfracarea'])
            tmpdens   = hpdens[mainreg & mask]
        
            tmpsyst   = hpdicttmp[syst][isreg]
            
            if percentiles is not None: 
                xlim = (np.percentile(tmpsyst[tmpsyst>0],(percentiles[0],percentiles[1])))
                
            
        #xlim      = tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))].min(), tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))].max()
        #xlim, _ = get_systplot(syst)
        #if clip: xlim = np.percentile(tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))],(1,99))
            tmpdens   = hpdens[isreg]
        
            if denslims:
                tmp = ((tmpdens/hpdicttmp['meandens_'+namesel+'_'+reg]>densmin) & 
                     (tmpdens/hpdicttmp['meandens_'+namesel+'_'+reg]<densmax) & 
                     (tmpsyst>xlim[0]) & 
                     (tmpsyst<xlim[1]))
            else:
                tmp       = (tmpsyst>xlim[0]) & (tmpsyst<xlim[1])
            
            systquant = tmpsyst[tmp] #systematics per region
            systdens  = tmpdens[tmp] #target density per region per bit
        
            if overbyreg: 
                if (reg=='all'): systdens /= hpdicttmp['meandens_'+namesel+'_'+'all']
                else: systdens /= hpdicttmp['meandens_'+namesel+'_'+reg] #density/mean density per bit per region
            else: systdens /= hpdicttmp['meandens_'+namesel+'_'+'all'] #density/mean density per bit overall desi footprint
        
        #print(systdens)
        #print(systquant)
        # get eta / eta_mean in nx bins
            plotxgrid, systv, systverr, xgrid = pixcorr(x=systquant, y=systdens, nx=nx, xlim=xlim)
        
            if label:
                if reg == 'south': lab = 'DECaLS+DES'+' & '+namesel
                elif reg == 'north': lab = 'BASS/MzLS'+' & '+namesel
                else: lab = reg+' & '+namesel
            else: lab = None
            
            if (len(regs) < 2) & (len(namesel) < 2): newcol, lw, alpha = 'k', 1, 1.0
            else: newcol = col
        
            if weights:
            
                if ws is None:
                    b0, m0 = findlinmb(plotxgrid, systv, systverr)
                    ws0 = 1./(m0*systquant+b0)
                    ax.text(plotxgrid[2], 1.18, r'b = %2.3g' %(b0))
                    ax.text(plotxgrid[2], 1.16, r'm = %2.3g' %(m0))
                else: 
                    ws0 = ws[isreg]
                    ws0 = ws0[tmp]
                
                plotxgrid_w, systv_w, systverr_w, xgrid_w = pixcorr(x=systquant, y=systdens*ws0, nx=nx, xlim=xlim)
            
                if label: labw = lab+' weighted'
                else: labw = None
            
                if not onlyweights: ax.errorbar(plotxgrid, systv, systverr, color=newcol, ecolor=newcol, zorder=1, lw=2*lw, alpha=alpha, label=lab)
                ax.errorbar(plotxgrid_w, systv_w, systverr_w, color=newcol, ecolor=newcol, zorder=1, lw=2*lw, ls='--',alpha=alpha, label=labw)
            else: 
                ax.errorbar(plotxgrid,systv,systverr,color=newcol,ecolor=newcol,zorder=1,lw=2*lw,alpha=alpha, label=lab)
            
            # histogram
            height,_ = np.histogram(systquant,bins=xgrid)
            height   = height.astype(float) / 1.e4
            xcent    = 0.5*(xgrid[1:]+xgrid[:-1])
            if (reg=='all') or (len(regs) < 2):
                axh.bar(xcent,height,align='center',width=xwidth,alpha=0.3,color=newcol)
            elif (len(regs) > 1): axh.step(xcent,height,where='mid',alpha=alpha,lw=lw,color=newcol)
        
            if label: ax.legend()
        
            if reg == 'all': x,yall = plotxgrid,systv
            elif reg == 'north': ynorth = systv
            elif reg == 'decals': ydecals = systv
            elif reg == 'des': ydes = systv
            
    if (weights) & (ws is None):
        return b0, m0
    elif get_values: 
        return plotxgrid,systv,systverr, lab
    else:
        return ax
    #return x, yall, ynorth, ydecals, ydes
    
def pixcorr(x=None, y=None, nx=20, xlim=None):
    
    xgrid = xlim[0]+np.arange(nx+1)/float(nx)*(xlim[1]-xlim[0])
    plotxgrid    = (xgrid[0:-1]+xgrid[1:])/2.
    systnobj     = np.ones(nx)*float('NaN')
    systv        = np.ones(nx)*float('NaN')
    systverr     = np.ones(nx)*float('NaN')
    for j in range(nx):
        tmp      = np.where((x >= xgrid[j]) & (x < xgrid[j+1]))[0]
        systnobj[j]= len(tmp)
        if (len(tmp) > 0):
            systv[j]   = np.mean(y[tmp])
            systverr[j]= np.std(y[tmp])/np.sqrt(len(y[tmp]))
    
    return plotxgrid, systv, systverr, xgrid
    
def findlinmb(x, y, yerr):
    #finds linear fit parameters
    lf = linfit(x,y,yerr)
    inl = np.array([1.,0])
    b0,m0 = optimize.fmin(lf.chilin,inl, disp=False)
    return b0,m0

class linfit:
    def __init__(self,xl,yl,el):
        self.xl = xl
        self.yl = yl
        self.el = el
              
    def chilin(self,bml):
        chi = 0
        b = bml[0]
        m = bml[1]
        for i in range(0,len(self.xl)):
            y = b+m*self.xl[i]
            chi += (self.yl[i]-y)**2./self.el[i]**2.
        return chi
    
def mollweide(hpdict=None, C=None, namesel=None, reg=None, nside=256, projection=None, n=None, org=None, cm=None, 
              fig=None, gs=None, ws=None, perc=(1, 99), title=None, cval=None, desifootprint=True):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)
    
    if desifootprint: isdesi = hpdict['isdesi']
    else: isdesi = np.ones_like(hpdict['ra'], dtype=bool)
        
    if reg == 'all': mainreg = (isdesi) & (hpdict['bgsfracarea']>0)
    else: mainreg = (isdesi) & (hpdict['bgsfracarea']>0) & (hpdict['is'+reg])
        
    ramw,decmw = get_radec_mw(hpdict['ra'],hpdict['dec'],org)
    if C is None:
        hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
        hpmean = hpdict['meandens_'+namesel+'_'+reg]
        clab      = 'LS dr8/'+namesel+r' density [deg$^{-2}$]'
    else:
        key = list(C.keys())[0]
        val = C[key]
        hpdens = val
        hpmean = np.mean(val)
        clab = key
    
    if ws is not None: hpdens = hpdens*ws
    
    xlims = np.percentile(hpdens[np.isfinite(hpdens)], perc)
    width = (xlims[1] - xlims[0])/2.
    #cmin,cmax = (0.1*hpmean,2*hpmean)
    if cval is None: cmin,cmax = (hpmean-width,hpmean+width)
    else: cmin, cmax = cval[0], cval[1]
    cbarticks = np.linspace(cmin,cmax,5)
    cbar_ylab = ['%.0f' % x for x in cbarticks]
        
    # density skymap + hist
    # mollweide
    
    ax     = plt.subplot(gs[n],projection=projection)
    if title is not None: ax.set_title(title, size=18)
    _      = set_mwd(ax,org=org)
    SC  = ax.scatter(ramw[mainreg],decmw[mainreg],s=1,
        c=hpdens[mainreg],
        cmap=cm,vmin=cmin,vmax=cmax,rasterized=True)
    p  = ax.get_position().get_points().flatten()
    cax= fig.add_axes([p[0]+0.2*(p[2]-p[0]),p[1]+0.2*(p[3]-p[1]),0.3*(p[2]-p[0]),0.025])
    cbar = plt.colorbar(SC, cax=cax, orientation='horizontal', ticklocation='top', extend='both', ticks=cbarticks)
    cbar.set_label(clab,fontweight='bold')
    cbar.ax.set_yticklabels(cbar_ylab)
    
def mollweideOLD(hpdict=None, namesel=None, reg=None, nside=256, projection=None, n=None, org=None, cm=None, fig=None, gs=None):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)
    
    if reg == 'all': mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
    else: mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0) & (hpdict['is'+reg])
    ramw,decmw = get_radec_mw(hpdict['ra'],hpdict['dec'],org)
    hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
    hpmean = hpdict['meandens_'+namesel+'_'+reg]
    cmin,cmax = (0.1*hpmean,2*hpmean)
    cbarticks = np.linspace(cmin,cmax,5)
    cbar_ylab = ['%.0f' % x for x in cbarticks]
    clab      = 'LS dr8/'+namesel+r' density [deg$^{-2}$]'

    # density skymap + hist
    # mollweide
    ax     = plt.subplot(gs[n],projection=projection)
    _      = set_mwd(ax,org=org)
    SC  = ax.scatter(ramw[mainreg],decmw[mainreg],s=1,
        c=hpdens[mainreg],
        cmap=cm,vmin=cmin,vmax=cmax,rasterized=True)
    p  = ax.get_position().get_points().flatten()
    cax= fig.add_axes([p[0]+0.2*(p[2]-p[0]),p[1]+0.2*(p[3]-p[1]),0.3*(p[2]-p[0]),0.025])
    cbar = plt.colorbar(SC, cax=cax, orientation='horizontal', ticklocation='top', extend='both', ticks=cbarticks)
    cbar.set_label(clab,fontweight='bold')
    cbar.ax.set_yticklabels(cbar_ylab)
    
def pixhistregs(hpdict=None, namesel=None, regs=None, cols=None, nside=256, n=None, fig=None, gs=None, primary=None):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)
    
    hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
    ax     = plt.subplot(gs[n])
    
    clab      = 'LS dr8/'+namesel+r' density [deg$^{-2}$]'
    hpmean = hpdict['meandens_'+namesel+'_'+'all']
    cmin,cmax = (0.1*hpmean,2*hpmean)
    xgrid  = np.linspace(cmin,cmax,51)
    
    # hist
    if len(regs) < 2:
        mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0) & (hpdict['is'+regs[0]])
        ax.hist(hpdens[mainreg],bins=xgrid,histtype='stepfilled',alpha=0.3,color='k',density=True, 
                label='dr8/'+reg+' ('+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+')')
    else:
        for reg, col in zip(regs, cols):
            if reg == 'all': mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
            else: mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0) & (hpdict['is'+reg])
            if reg == primary:
                ax.hist(hpdens[mainreg], bins=xgrid,histtype='stepfilled',alpha=0.3,color='k',lw=2,density=True,
                    label='dr8/'+reg+' ('+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+')')
            else:
                ax.hist(hpdens[mainreg], bins=xgrid,histtype='step',alpha=0.8,color=col,lw=2,density=True,
                    label='dr8/'+reg+' ('+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+')')
            
    ax.set_xlabel(clab,fontsize=15)
    ax.set_ylabel('norm. counts',fontsize=15)
    ax.set_xlim(cmin,cmax)
    ax.grid(True)
    ax.legend(loc='upper right')
    
def pixhistnamesels(hpdict=None, namesels=None, reg=None, cols=None, nside=256, n=None, fig=None, gs=None, primary=None, xlab=None):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)
    
    if reg == 'all': mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
    else: mainreg = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0) & (hpdict['is'+reg])
    ax     = plt.subplot(gs[n])
    
    if xlab is not None: clab = 'LS dr8 '+xlab+r' density [deg$^{-2}$]'
    else: clab = 'LS dr8 '+reg+r' density [deg$^{-2}$]'
    hpmean = hpdict['meandens_'+'any'+'_'+reg]
    cmin,cmax = (0.1*hpmean,2*hpmean)
    xgrid  = np.linspace(cmin,cmax,51)
    
    # hist
    if len(namesels) < 2:
        hpdens = (hpdict['south_n'+namesel[0]] + hpdict['north_n'+namesel[0]] ) / (pixarea * hpdict['bgsfracarea'])
        ax.hist(hpdens[mainreg],bins=xgrid,histtype='stepfilled',alpha=0.3,color='k',density=True, 
                label='dr8/'+namesels[0]+' ('+'%.0f'%hpdict['meandens_'+namesel[0]+'_'+reg]+')')
    else:
        for namesel, col in zip(namesels, cols):
            hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
            if namesel == primary:
                ax.hist(hpdens[mainreg], bins=xgrid,histtype='stepfilled',alpha=0.3,color='k',lw=2,density=True,
                    label='dr8/'+namesel+' ('+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+')')
            else:
                ax.hist(hpdens[mainreg], bins=xgrid,histtype='step',alpha=0.8,color=col,lw=2,density=True,
                    label='dr8/'+namesel+' ('+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+')')
            
    ax.set_xlabel(clab,fontsize=15)
    ax.set_ylabel('norm. counts',fontsize=15)
    ax.set_xlim(cmin,cmax)
    ax.grid(True)
    ax.legend(loc='upper right')
    
# https://matplotlib.org/examples/api/colorbar_only.html
def mycmap(name,n,cmin,cmax):
	cmaporig = matplotlib.cm.get_cmap(name)
	mycol    = cmaporig(np.linspace(cmin, cmax, n))
	cmap     = matplotlib.colors.ListedColormap(mycol)
	cmap.set_under(mycol[0])
	cmap.set_over (mycol[-1])
	return cmap

def overdensity(cat, star, radii_1, nameMag, slitw, density=False, magbins=(8,14,4), radii_2=None, 
                grid=None, SR=[2, 240.], scaling=False, nbins=101, SR_scaling=4, logDenRat=[-3, 3], 
                    radii_bestfit=True, annulus=None, bintype='2', filename=None, log=False):
    '''
    Get scatter and density plots of objects of cat1 around objects of cat2 within a search radius in arcsec.

    Inputs
    ------
    cat: (array) catalogue 1;
    star: (array) catalogue 2;
    nameMag: (string) label of magnitude in catalogue 2;
    slitw: (float, integer) slit widht;
    density: (boolean) True to get the density as function of distance (arcsec) within shells;
    magbins: (integers) format to separate the magnitude bins in cat2 (min, max, number bins);

    Output
    ------
    (distance (arcsec), density) if density=True
    '''
    from io_ import search_around #if issues with this, comment it run notebook and then uncomment it and run notebook again
    # define the slit width for estimating the overdensity off diffraction spikes
    slit_width = slitw
    search_radius = SR[1]

    # Paramater for estimating the overdensities
    annulus_min = SR[0]
    annulus_max = SR[1]

    ra2 = star['RA']
    dec2 = star['DEC']
    
    ra1 = cat['RA']
    dec1 = cat['DEC']

    if density:

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2, dec2, ra1, dec1,
                                                 search_radius=search_radius)
        density = []
        shells = np.linspace(1, search_radius, search_radius)
        for i in range(len(shells)-1):

            ntot_annulus = np.sum((d2d>shells[i]) & (d2d<shells[i+1]))
            density_annulus = ntot_annulus/(np.pi*(shells[i+1]**2 - shells[i]**2))
            bincenter = (shells[i]+shells[i+1])/2

            density.append([bincenter, density_annulus])

        density = np.array(density).transpose()
        plt.figure(figsize=(12, 8))
        plt.semilogy(density[0], density[1])
        plt.xlabel(r'r(arcsec)')
        plt.ylabel(r'N/($\pi r^2$)')
        plt.grid()
        plt.show()

        return density


    if bintype == '2':
        mag_bins = np.linspace(magbins[0], magbins[1], magbins[2]+1)
        mag_bins_len = len(mag_bins)-1
    elif bintype == '1':
        mag_bins = np.linspace(magbins[0], magbins[1], magbins[2])
        mag_bins_len = len(mag_bins)
    elif bintype == '0':
        mag_bins = np.array(magbins)
        mag_bins_len = len(mag_bins)-1
    else:
        raise ValueError('Invaid bintype. Choose bintype = 0, 1, 2')
    print('mag_bins_len:', mag_bins_len)
    
    
    if grid is not None:
        rows, cols = grid[0], grid[1]
    else:
        rows, cols = len(mag_bins), 1
    figsize = (8*cols, 8*rows)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.3, hspace=0.2)
    fig = plt.figure(num=1, figsize=figsize)
    ax = []
        
    for index in range(mag_bins_len):
        if (bintype == '2') or (bintype == '0'):
            mask_star = (star[nameMag]>mag_bins[index]) & (star[nameMag]<mag_bins[index+1])
            title = '{:.2f} < {} < {:.2f}'.format(mag_bins[index], nameMag, mag_bins[index+1], np.sum(mask_star))    
        elif bintype == '1':
            if index==0:
                mask_star = (star[nameMag]<mag_bins[index])
                title = '{} < {:.2f}'.format(nameMag,mag_bins[0], np.sum(mask_star))
            else:
                mask_star = (star[nameMag]>mag_bins[index-1]) & (star[nameMag]<mag_bins[index])
                title = '{:.2f} < {} < {:.2f}'.format(mag_bins[index-1], nameMag, mag_bins[index], np.sum(mask_star))
        else:
            raise ValueError('Invaid bintype. Choose bintype = 0, 1, 2')

        if log: print(title)
        if bintype != '1':
            magminrad = circular_mask_radii_func([mag_bins[index+1]], radii_1, bestfit=radii_bestfit)[0]
            magmaxrad = circular_mask_radii_func([mag_bins[index]], radii_1, bestfit=radii_bestfit)[0]
        else:
            if index == 0:
                magminrad = circular_mask_radii_func([mag_bins[index]], radii_1, bestfit=radii_bestfit)[0]
                magmaxrad = magminrad
            else:
                magminrad = circular_mask_radii_func([mag_bins[index]], radii_1, bestfit=radii_bestfit)[0]
                magmaxrad = circular_mask_radii_func([mag_bins[index-1]], radii_1, bestfit=radii_bestfit)[0]

        if log: print('MARK #1')
            
        if not scaling:
            #get the mask radii from the mean magnitude
            mag_mean = np.mean(star[nameMag][mask_star])
            if log: print('mag_mean', mag_mean)
            mask_radius = circular_mask_radii_func([mag_mean], radii_1, bestfit=radii_bestfit)[0]
            if radii_2:
                mask_radius2 = circular_mask_radii_func([mag_mean], radii_2, bestfit=radii_bestfit)[0]
                if log: print('mask_radius2', mask_radius2)

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2[mask_star], dec2[mask_star], ra1, dec1,
                                                 search_radius=annulus_max, verbose=False)
        
        if log: print('MARK #2')

        Nsources = len(ra2[mask_star])
        perc_sources = 100*len(ra2[mask_star])/len(ra2)
        
        #print('%d sources ~%g %% ' %(Nsources, perc_sources))
        
        mag_radii = circular_mask_radii_func(star[nameMag][mask_star][idx2], radii_1, bestfit=radii_bestfit)
        #print(len(d_ra), len(mag_radii))
        if log: print('mag_radii MAX:',mag_radii.max(), 'mag_radii MIN:',mag_radii.min())
        if log: print('mag MAX:',star[nameMag][mask_star][idx2].max(), 'mag MIN:',star[nameMag][mask_star][idx2].min())

        #markersize = np.max([0.01, np.min([10, 0.3*100000/len(idx2)])])
        #axis = [-search_radius*1.05, search_radius*1.05, -search_radius*1.05, search_radius*1.05]
        #axScatter = scatter_plot(d_ra, d_dec, markersize=markersize, alpha=0.4, figsize=6.5, axis=axis, title=title)
        
        row = (index // cols)
        col = index % cols
        ax.append(fig.add_subplot(gs[row, col]))
        
        if scaling:
            d2d_arcsec = d2d
            #d2d is already in arcsec with r^2 = ra^2*cos(dec)^2 + dec^2
            d_ra, d_dec, d2d = d_ra/mag_radii, d_dec/mag_radii, d2d_arcsec/mag_radii
            search_radius = SR_scaling #d2d.max() - d2d.max()*0.3
            #ntot_annulus = np.sum((d2d_arcsec>annulus_min) & (d2d<search_radius))
            ntot_annulus = np.sum(d2d<search_radius)
            #density_annulus = ntot_annulus/(np.pi*(search_radius**2 - d2d[d2d_arcsec > 2].min()**2))
            density_annulus = ntot_annulus/(np.pi*(search_radius**2))
            #print('ntot_annulus:', ntot_annulus, 'density_annulus:', density_annulus)
            if log: print('d2d min=%2.3g, d2d max=%2.3g' %(d2d.min(), d2d.max()))
        else:
            d2d_arcsec = None
            ntot_annulus = np.sum((d2d>annulus_min) & (d2d<annulus_max))
            density_annulus = ntot_annulus/(np.pi*(annulus_max**2 - annulus_min**2))
        
        if annulus is not None:
            annMask = np.ones(len(cat), dtype='?')
            d_ra2 = np.zeros(len(cat))
            d_dec2 = np.zeros(len(cat))
            d2d2 = np.zeros(len(cat))
            d_ra2[idx1] = d_ra
            d_dec2[idx1] = d_dec
            d2d2[idx1] = d2d
            if log: print(len(cat), len(d_ra2), len(d_dec2))
            #print(len(set(idx1)), len(set(idx2)))
            #print(idx1.max(), idx2.max())
            #angle_array = np.linspace(0, 2*np.pi, 240)
            annMask &= np.logical_and((d_ra2**2 + d_dec2**2) < annulus[1]**2, (d_ra2**2 + d_dec2**2) > annulus[0]**2)
            
            #annMask &= np.logical_and(d_dec < annulus[1] * np.cos(angle_array), d_dec > annulus[0] * np.cos(angle_array))
        
        if scaling:
            mask_radius = None
        
        bins, mesh_d2d, density_ratio = relative_density_plot(d_ra, d_dec, d2d, search_radius,
                        ref_density=density_annulus, return_res=True, show=False, nbins=nbins, 
                            ax=ax[-1], d2d_arcsec=d2d_arcsec, annulus_min=annulus_min, 
                                logDenRat=logDenRat, mask_radius=magmaxrad, log=log)
   
        if not scaling:
            angle_array = np.linspace(0, 2*np.pi, 240)
            for i in [magminrad, magmaxrad]:
                x = i * np.sin(angle_array)
                y = i * np.cos(angle_array)
                ax[-1].plot(x, y, 'k', lw=2, alpha=0.4)
            ax[-1].plot(mask_radius * np.sin(angle_array), mask_radius * np.cos(angle_array), 'k', lw=2)
                
            
            ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.05, '%d sources ~%2.3g %% ' %(Nsources, perc_sources), fontsize=10,color='k')
            ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.11, '%d objects ~%2.3g %% ' %(ntot_annulus, 100*ntot_annulus/len(ra1)), fontsize=10,color='k')
            #ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.17, '$\eta$=%2.3g arcsec$^{-2}$' %(density_annulus), fontsize=8,color='k')

            ax[-1].set_xlabel(r'$\Delta$RA (arcsec)', size=20)
            ax[-1].set_ylabel(r'$\Delta$DEC (arcsec)', size=20)
        
            if radii_2:
                x2 = mask_radius2 * np.sin(angle_array)
                y2 = mask_radius2 * np.cos(angle_array)
                ax[-1].plot(x2, y2, 'gold', lw=2, linestyle='-')
        else:
            angle_array = np.linspace(0, 2*np.pi, 100)
            x = 1 * np.sin(angle_array)
            y = 1 * np.cos(angle_array)
            ax[-1].plot(x, y, 'k', lw=2)
            
            ax[-1].text(-SR_scaling+0.1, SR_scaling-0.2, '%d sources ~%2.3g %% ' %(Nsources, perc_sources), fontsize=12,color='k')
            ax[-1].text(-SR_scaling+0.1, -SR_scaling+0.1, '%d objects ~%2.3g %% ' %(ntot_annulus, 100*ntot_annulus/len(ra1)), fontsize=12,color='k')
            #ax[-1].text(-SR_scaling+0.1, SR_scaling-0.9, '$\eta$=%2.3g deg$^{-2}$' %(density_annulus), fontsize=8,color='k')

            ax[-1].set_xlabel(r'$\Delta$RA/$R_{BS}$', size=18)
            
            ybox1 = TextArea(r'$\Delta$DEC/$R_{BS}$, ', textprops=dict(color="k", size=18,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea(r'$\log_{2}(\eta (\Delta R)/\bar{\eta})$', textprops=dict(color="r", size=18,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.25), 
                                              bbox_transform=ax[-1].transAxes, borderpad=0.)

            #if ((cols > 1) & (index == 0)) or (cols < 2): ax[-1].set_ylabel(r'$\Delta$DEC/$R_{BS}$', size=18)
            if ((cols > 1) & (index == 0)) or (cols < 2): ax[-1].add_artist(anchored_ybox)
            
        ax[-1].set_title(title, size=18)
        ax[-1].axvline(0, ls='--', c='k')
        ax[-1].axhline(0, ls='--', c='k')
        if annulus is not None:
            for i in annulus:
                x = i * np.sin(angle_array)
                y = i * np.cos(angle_array)
                ax[-1].plot(x, y, 'yellow', lw=3, ls='-')
                
    if filename is not None:
            fig.savefig(filename+'.png', bbox_inches = 'tight', pad_inches = 0)
    
    if annulus is not None:
        return d2d2, d_ra2, d_dec2, annMask

def relative_density_plot(d_ra, d_dec, d2d, search_radius, ref_density, nbins=101, return_res=False, 
                          show=True, ax=plt, d2d_arcsec=None, annulus_min=2, logDenRat=[-3,3], 
                              mask_radius=None, log=False):

    bins = np.linspace(-search_radius, search_radius, nbins)
    bin_spacing = bins[1] - bins[0]
    bincenter = (bins[1:]+bins[:-1])/2
    mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
    mesh_d2d = np.sqrt(mesh_ra**2 + mesh_dec**2)
    if d2d_arcsec is not None:
        mask = (d2d_arcsec>annulus_min) #to avoid self match with stars
    else:
        mask = (d2d>annulus_min) #to avoid self match with stars
    #taking the 2d histogram and divide by the area of each bin to get the density
    density, _, _ = np.histogram2d(d_ra[mask], d_dec[mask], bins=bins)/(bin_spacing**2)
    #ignoring data outside the circle with radius='search radius'
    #print('Nbins:',len(bins), 'binArea:', bin_spacing**2, 'Nobjects:', len(d_ra[mask]))
    #pix_density = len(d_ra[mask])/((len(bins)**2)*(bin_spacing**2))
    #print('tot_density_pix:', pix_density)
    
    #mean density at search radius
    if search_radius < 10:
        meanmask = np.logical_and(mesh_d2d <= search_radius, mesh_d2d > 1.2)
    else:
        meanmask = np.logical_and(mesh_d2d <= search_radius, mesh_d2d > 100.)
    ref_density = np.mean(density[meanmask])
    
    #density profile
    dist = np.linspace(0., search_radius, nbins)
    dist2 = np.linspace(0.008, search_radius, nbins/2.)
    dist_spacing = dist[1] - dist[0]
    dist_spacing2 = dist2[1] - dist2[0]
    dpx, dpy, dpx2, dpy2 = [], [], [], []
    for i, j in enumerate(dist):
        #for the cumulative radia profile
        dmask = mesh_d2d <= j
        drcumu = np.log2(np.mean(density[dmask]/ref_density))
        if drcumu is np.nan:
            dpy.append(-1)
        else:
            dpy.append(drcumu)
        dpx.append(j)
    for i, j in enumerate(dist2[:-1]):
        #for the no cumulative radia profile
        dmask2 = np.logical_and(mesh_d2d < dist2[i+1], mesh_d2d >= dist2[i])
        drnocumu = np.log2(np.mean(density[dmask2]/ref_density))
        #drnocumu = np.mean(density[dmask2]/ref_density) -1.
        if drnocumu is np.nan:
            dpy2.append(-1)
        else:
            dpy2.append(drnocumu)
        dpx2.append(dist2[i] + dist_spacing2/2.)
    
    if search_radius < 10:
        dpy = np.array(dpy)
        dpy2 = np.array(dpy2)
        dmax = dpy2[np.array(dpx2) > 1].max()
        dmin = dpy2[np.array(dpx2) > 1].min()
        maglimrad = 1
    else:
        dpy20 = np.array(dpy2)
        dpy = np.array(dpy)*search_radius/logDenRat[1]
        dpy2 = np.array(dpy2)*search_radius/logDenRat[1]
        dmax = dpy20[np.array(dpx2) > mask_radius].max()
        dmin = dpy20[np.array(dpx2) > mask_radius].min()
        maglimrad = mask_radius
    
    if log: print('density cumu (min, max): (%2.3g, %2.3g)' %(np.array(dpy).min(), np.array(dpy).max()))
    if log: print('density non-cumu (min, max): (%2.3g, %2.3g)' %(np.array(dpy2).min(), np.array(dpy2).max()))
    
    mask = mesh_d2d >= bins.max()-bin_spacing
    density[mask] = np.nan
    #density_ratio = density/ref_density
    density_ratio = np.log2(density/ref_density)
    
    idxinf = np.where(np.isinf(density_ratio))
    #print('inf values:',density_ratio[idxinf])
    if log: print('%d of inf in density ratio out of a total of %d' %(len(density_ratio[idxinf]), len(density_ratio[~np.isnan(density_ratio)])))
    density_ratio[idxinf] = logDenRat[0]
    #print('inf values AFTER:',density_ratio[idxinf])
    
    den_rat = density_ratio[~np.isnan(density_ratio)]
    denmin = den_rat.min()
    denmax = den_rat.max()
    if log: print('Minimum density ratio = %g, Maximum density ratio = %g' %(denmin, denmax))
    if log: print('----------------')
    fig = plt.figure(1)
    #img = ax.imshow(density_ratio.transpose()-1, origin='lower', aspect='equal',
    img = ax.imshow(density_ratio.transpose(), origin='lower', aspect='equal',
               cmap='seismic', extent=bins.max()*np.array([-1, 1, -1, 1]), vmin=logDenRat[0], vmax=logDenRat[1])
    #ax.colorbar(fraction=0.046, pad=0.04)
    cb = fig.colorbar(img, fraction=0.046, pad=0.04)
    
    cb.set_label(label=r'$\log_{2}(\eta_{pix}/\bar{\eta})$', weight='bold', size=18)
    cb.ax.tick_params(labelsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    #ax.plot(np.array(dpx), dpy, lw=2.5, color='green')
    ax.plot(np.array(dpx2), dpy2, lw=2, color='darkred')
    
    # find the max, min of density ratio profile for distances > 1
    ax.text(1*search_radius/10., search_radius - 2*search_radius/30, '$max(\eta(\Delta R)/\eta, r>%i)=%2.3g$' %(maglimrad, 2**(dmax)), fontsize=14,color='k')
    #ax.text(4*search_radius/10., search_radius - 4*search_radius/30, '$min(\eta(\delta r)/\eta)=%2.3g$' %(2**(dmin)), fontsize=10,color='k')
    
    ax.set_ylim(-search_radius, search_radius)
    if show:
        ax.show()

    if return_res:
        return bins, mesh_d2d, density_ratio
    
def circular_mask_radii_func(MAG, radii_1, bestfit=True):
    '''
    Define mask radius as a function of the magnitude

    Inputs
    ------
    MAG: Magnitude in (array);
    radii_1: radii as a function of magnitude;
    bestfit: True to get the best-fit of radii instead;

    Output
    ------
    radii: mask radii (array)
    '''
    
    x, y = np.transpose(radii_1)
    
    if not bestfit:
        circular_mask_radii_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    else:
        powr = np.polyfit(x, y, 2)
        Y = powr[0]*x**2 + powr[1]*x + powr[2]
        circular_mask_radii_func = interp1d(x, Y, bounds_error=False, fill_value=(y[0], y[-1]))
        
    # mask radius in arcsec
    return circular_mask_radii_func(MAG)

def scatterplot(coord=None, catmask=None, xlim=None, ylim=None, title=None, fig=None, gs=None, n=None, ylab=True, 
                xlab=True, hline=None, vline=None, fmcline=False, file=None, contour1=None):
    
    x, y = coord.keys()
    
    ax = fig.add_subplot(gs[n])
    if title is not None: ax.set_title(r'%s' %(title), size=20)
    if xlim is None: xlim = limits()[x]
    if ylim is None: ylim = limits()[y]
    masklims = (coord[x] > xlim[0]) & (coord[x] < xlim[1]) & (coord[y] > ylim[0]) & (coord[y] < ylim[1])
    
    if catmask is None: 
        keep = masklims
        ax.scatter(coord[x][keep], coord[y][keep])
    else:
        for key, val in zip(catmask.keys(), catmask.values()):
            keep = (val) & (masklims)
            
            if np.sum(keep) < 100: s = 15
            elif np.sum(keep) > 1000: s = 1
            else: s = 3
        
            ax.scatter(coord[x][keep], coord[y][keep], s=s, label=key)
            
    if ylab: ax.set_ylabel(r'%s' %(y), size=20)
    if xlab: ax.set_xlabel(r'%s' %(x), size=20)
    if hline is not None: ax.axhline(hline, ls='--', lw=2, c='r')
    if vline is not None: ax.axvline(vline, ls='--', lw=2, c='r')
        
    lgnd = plt.legend()
    [handle.set_sizes([20]) for handle in lgnd.legendHandles]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #contour galaxies
    #np.logspace(0, 2, 5)
    if contour1 is not None:
        bgs_den = density_patch(coord=contour1, xlim=xlim, ylim=ylim, plot=False, nmin=0)
        ax.contour(bgs_den.transpose(), levels=(0, 10), origin='lower', aspect='equal',
              extent=np.array([xlim[0], xlim[1], xlim[0], xlim[1]]), colors='black', alpha=0.3)
        
    if file is not None:
        fig.savefig(file+'.png', bbox_inches = 'tight', pad_inches = 0)
        
def limits():
    
    limits = {}
    limits['Grr'] = (-3, 5)
    limits['G-rr'] = (-3, 5)
    limits['g-z'] = (-2, 6)
    limits['r'] = (15, 20.1)
    limits['rfibmag'] = (16, 27)
    limits['g-r'] = (-2, 6)
    limits['r-z'] = (-0.7, 2.8)
    
    return limits

def plot_venn3(A, B, C, norm=None, labels=None, file=None, title=None, colors=None):
    '''inputs A, B, C must booleans.'''
    
    A1 = A
    B1 = B
    C1 = C
    AB = (A1) & (B1)
    AC = (A1) & (C1)
    BC = (B1) & (C1)
    ABC = (A1) & (B1) & (C1)
            
    if norm is None: norm, sf = 1, 1
    else: sf = 1
        
    a1 = round((np.sum(A1) - np.sum(AB) - np.sum(AC) + np.sum(ABC))/norm, sf)
    a2 = round((np.sum(B1) - np.sum(AB) - np.sum(BC) + np.sum(ABC))/norm, sf)
    a3 = round((np.sum(AB) - np.sum(ABC))/norm, sf)
    a4 = round((np.sum(C1) - np.sum(AC) - np.sum(BC) + np.sum(ABC))/norm, sf)
    a5 = round((np.sum(AC) - np.sum(ABC))/norm, sf)
    a6 = round((np.sum(BC) - np.sum(ABC))/norm, sf)
    a7 = round(np.sum(ABC)/norm, sf)
        
    if labels is None: labels = ['Group A', 'Group B', 'Group C']
    if colors is None: colors = ['r', 'green', 'b']
        
    fig = plt.figure(figsize=(7,7))
    v=venn3([a1, a2, a3, a4, a5, a6, a7], set_labels = (labels[0], labels[1], labels[2]), set_colors=(colors), alpha = 0.6)
    c=venn3_circles([a1, a2, a3, a4, a5, a6, a7], linestyle='dotted', linewidth=1, color="k")
    #c[1].set_lw(1.0)
    #if colors is not None:
        #c[0].set_color(colors[0])
        #c[1].set_color(colors[1])
        #c[2].set_color(colors[2])
    #c[0].set_ls('dotted')
    #c[1].set_ls('solid')
    #c[2].set_ls('dashed')
    #c[1].set_lw(2.0)
    
    if title is not None: plt.title(title, size=20)
    if file is not None:
        fig.savefig(file+'.png', bbox_inches = 'tight', pad_inches = 0)

    plt.show()
    
    
from matplotlib_venn import venn2, venn2_circles

def plot_venn2(A,B,area, title=None, labels=None, savefile=None):

    fig = plt.figure(figsize=(10,10))

    sf = 2
    #area = hpdict0['bgsarea_'+survey]

    a = A
    b = B
    c = (a) & (b)

    a1 = round(((np.sum(a) - np.sum(c))/area), sf)
    b1 = round(((np.sum(b) - np.sum(c))/area), sf)
    c1 = round(np.sum(c)/area, sf)

    if title is not None: plt.title(title, size=18)
    if labels is None: labels = ('group A', 'group B')

    venn2([a1, b1, c1], set_labels = labels)
    #venn2([a1, b1, c1])
    c=venn2_circles([a1, b1, c1], linestyle='solid', linewidth=1, color="k")

    #filename = '%s/%s' %(pathdir, savefile)
    if savefile is not None:
        fig.savefig(savefile+'.png', bbox_inches = 'tight', pad_inches = 0)
    

def scatterplot(coord=None, catmask=None, xlim=None, ylim=None, title=None, fig=None, gs=None, n=None, ylab=True, 
                xlab=True, hline=None, vline=None, fmcline=False, file=None, contour1=None):
    
    x, y = coord.keys()
    
    ax = fig.add_subplot(gs[n])
    if title is not None: ax.set_title(r'%s' %(title), size=20)
    if xlim is None: xlim = limits()[x]
    if ylim is None: ylim = limits()[y]
    masklims = (coord[x] > xlim[0]) & (coord[x] < xlim[1]) & (coord[y] > ylim[0]) & (coord[y] < ylim[1])
    
    if catmask is None: 
        keep = masklims
        ax.scatter(coord[x][keep], coord[y][keep])
    else:
        for key, val in zip(catmask.keys(), catmask.values()):
            keep = (val) & (masklims)
            
            if np.sum(keep) < 100: s = 15
            elif np.sum(keep) > 1000: s = 1
            else: s = 3
        
            ax.scatter(coord[x][keep], coord[y][keep], s=s, label=key)
            
    if ylab: ax.set_ylabel(r'%s' %(y), size=20)
    if xlab: ax.set_xlabel(r'%s' %(x), size=20)
    if hline is not None: ax.axhline(hline, ls='--', lw=2, c='r')
    if vline is not None: ax.axvline(vline, ls='--', lw=2, c='r')
    if fmcline: 
        x_N1 = np.linspace(15.5, 17.1, 4)
        ax.plot(x_N1, 2.9 + 1.2 + x_N1, color='r', ls='--', lw=2)
        x_N2 = np.linspace(17.1, 18.3, 4)
        ax.plot(x_N2, x_N2*0.+21.2, color='r', ls='--', lw=2)
        x_N3 = np.linspace(18.3, 20.1, 4)
        ax.plot(x_N3, 2.9 + x_N3, color='r', ls='--', lw=2)
        
    lgnd = plt.legend()
    [handle.set_sizes([20]) for handle in lgnd.legendHandles]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #contour galaxies
    #np.logspace(0, 2, 5)
    if contour1 is not None:
        bgs_den = density_patch(coord=contour1, xlim=xlim, ylim=ylim, plot=False, nmin=0)
        ax.contour(bgs_den.transpose(), levels=(0, 10), origin='lower', aspect='equal',
              extent=np.array([xlim[0], xlim[1], xlim[0], xlim[1]]), colors='black', alpha=0.3)
        
    if file is not None:
        fig.savefig(file+'.png', bbox_inches = 'tight', pad_inches = 0)
        
def density_patch(coord=None, xlim=None, ylim=None, nbins=100, plot=False, title=None, nmin=None):
    
    from matplotlib.colors import LogNorm
    
    x, y = coord.keys()
    
    if title is not None: ax.set_title(r'%s' %(title), size=20)
    if xlim is None: xlim = limits()[x]
    if ylim is None: ylim = limits()[y]
    masklims = (coord[x] > xlim[0]) & (coord[x] < xlim[1]) & (coord[y] > ylim[0]) & (coord[y] < ylim[1])
    
    bins = np.linspace(xlim[0], xlim[1], nbins)
    bin_spacing = bins[1] - bins[0]
    bincenter = (bins[1:]+bins[:-1])/2
    #mesh_x, mesh_y = np.meshgrid(bincenter, bincenter)
    
    xx, yy = coord[x][masklims], coord[y][masklims]
    #xx, yy = coord[x], coord[y]
    
    #taking the 2d histogram and divide by the area of each bin to get the density
    density, _, _ = np.histogram2d(xx, yy, bins=bins)
    if nmin is not None:
        density[density < nmin] = np.nan
    
    if plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(density.transpose(), origin='lower', aspect='equal',
               cmap='seismic', extent=np.array([xlim[0], xlim[1], ylim[0], ylim[1]]), norm=LogNorm()) #, vmin=-3, vmax=3
        #plt.imshow(density.transpose(),
        #       cmap='seismic', norm=LogNorm()) #, vmin=-3, vmax=3
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel(x, size=20)
        plt.ylabel(y, size=20)
    
    return density