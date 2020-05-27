import numpy as np
import matplotlib.pyplot as plt
import os
import random

import bokeh.plotting as bk
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ImageURL, CustomJS, Span, OpenURL, TapTool, Panel, Tabs
from bokeh.models.widgets import CheckboxGroup, CheckboxButtonGroup, RadioGroup
from bokeh.layouts import gridplot, row, column
from bokeh.io import curdoc, show


def coordtopix2(center, coord, size, scale):
    
    RA_pix = []
    DEC_pix = []
    for i in range(len(coord[0])):
        print('RA_pix: %f, DEC_pix: %f' %(size/2. + (round(center[0], 12)-round(coord[0][i], 12))*3600./scale,
                                         size/2. + (round(center[1], 12)-round(coord[1][i], 12))*3600./scale))
        print('NORMAL:RA_pix: %f, DEC_pix: %f' %(float(size/2) + (center[0]-coord[0][i])*float(3600/scale),
                                         float(size/2) + (center[1]-coord[1][i])*float(3600/scale)))
        ra_pix = size/2. + (round(center[0], 12)-round(coord[0][i], 12))*3600./scale
        dec_pix = size/2. + (round(center[1], 12)-round(coord[1][i], 12))*3600./scale
        RA_pix.append(ra_pix)
        DEC_pix.append(dec_pix)
    
    return RA_pix, DEC_pix

def coordtopix(center, coord, size, scale):
    
    RA_pix = []
    DEC_pix = []
    for i in range(len(coord[0])):
        d_ra = (center[0]-coord[0][i])*3600
        d_dec = (center[1]-coord[1][i])*3600
        if d_ra > 180*3600:
            d_ra = d_ra - 360.*3600
        elif d_ra < -180*3600:
            d_ra = d_ra + 360.*3600
        else:
            d_ra = d_ra
        d_ra = d_ra * np.cos(coord[1][i]/180*np.pi)
        
        ra_pix = size/2. + d_ra/scale
        dec_pix = size/2. + d_dec/scale
        RA_pix.append(ra_pix)
        DEC_pix.append(dec_pix)
    
    return RA_pix, DEC_pix

def coordtopix_2(center, coord, size, scale):
    
    RA_pix = []
    DEC_pix = []
    for i in range(len(coord[0])):
        d_ra = (center[0]-coord[0][i])*3600
        d_dec = (center[1]-coord[1][i])*3600
        if d_ra > 180*3600:
            d_ra = d_ra - 360.*3600
        elif d_ra < -180*3600:
            d_ra = d_ra + 360.*3600
        else:
            d_ra = d_ra
        d_ra = d_ra * np.cos(coord[1][i]/180*np.pi)
        
        ra_pix = size/2. + d_ra/scale
        dec_pix = size/2. - d_dec/scale
        RA_pix.append(ra_pix)
        DEC_pix.append(dec_pix)
        
    return RA_pix, DEC_pix

def disttopix(D, scale):
    '''
    D must be in arcsec...
    '''
    
    dpix = D/scale
    
    return dpix


def html_postages(cat, coord=None, idx=None, notebook=True, savefile=None, htmltitle='page', veto=None, grid=[2,2], m=4, radius=4/3600, comparison=False):
    
    if notebook: bk.output_notebook()
    if savefile is not None:
        html_page = savefile + '.html'
        bk.output_file(html_page, title=htmltitle)
        print(html_page)

    
    plots = []
    sources = []
    layers = []
    tests = []
    
    RA, DEC = coord[0], coord[1]
    
    rows, cols = grid[0], grid[1]
    N = rows*cols
    scale_unit='pixscale'
    
    scale=0.262
    
    boxsize = 2*m*radius*3600
    size = int(round(boxsize/scale))
    print(boxsize, size)

    idx_list = random.sample(list(idx), rows*cols)
    
    layer_list = ['dr9f-south', 'dr9f-south-model', 'dr9f-south-resid', 'dr9g-south', 'dr9g-south-model', 'dr9g-south-resid',
                 'dr9f-north', 'dr9f-north-model', 'dr9f-north-resid', 'dr9g-north', 'dr9g-north-model', 'dr9g-north-resid']

#figlist = [figure(title='Figure '+str(i),plot_width=100,plot_height=100) for i in range(N)]

    if True:

        for num, idx in enumerate(idx_list):
    
            RAidx = RA[idx]
            DECidx = DEC[idx]
    
            ramin, ramax = RAidx-m*radius, RAidx+m*radius
            decmin, decmax = DECidx-m*radius, DECidx+m*radius
            dra = (ramax - ramin)/40
            ddec = (decmax - decmin)/40
            mask = (RA > ramin + dra) & (RA < ramax - dra) & (DEC > decmin + ddec) & (DEC < decmax - ddec)


            if comparison:
                
                TOOLTIPS = []
                for i in ['RA', 'DEC', 'morph', 'r', 'g', 'z', 'refcat']:
                    TOOLTIPS.append((i+'_g', '@'+i+'_g'))
                    TOOLTIPS.append((i+'_f', '@'+i+'_f'))
                
            else:
                
                TOOLTIPS = [
                    #("index", "$index"),
                    ("RA", "@RA"),
                    ("DEC", "@DEC"),
                    ("morph", "@morph"),
                    ("rmag", "@r"),
                    ("gmag", "@g"),
                    ("zmag", "@z"),
                    ("refcat", "@refcat"),
                    ]


            p = figure(plot_width=size, plot_height=size, tooltips=TOOLTIPS, tools="tap")
            p.axis.visible = False
            p.min_border = 0

            layers2 = []
            for layer in layer_list:
                
                source='http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%.12f&dec=%.12f&%s=%g&layer=%s&size=%g' % (RAidx, DECidx, scale_unit, scale, layer, size)
                url='http://legacysurvey.org/viewer-dev?ra=%.12f&dec=%.12f&layer=%s&zoom=15' %(RAidx, DECidx, layer)
                imfig_source = ColumnDataSource(data=dict(url=[source], txt=[source]))
                image1 = ImageURL(url="url", x=0, y=1, w=size, h=size, anchor='bottom_left')
                img_source = p.add_glyph(imfig_source, image1)
                
                layers2.append(img_source)

            taptool = p.select(type=TapTool)
            taptool.callback = OpenURL(url=url)

            colors = ['green', 'red', 'blue', 'cyan', 'yellow']
            circle_i = []
            #test_i = []
            for color, key, val in zip(colors, veto.keys(), veto.values()):

                ravpix, decvpix = coordtopix_2(center=[RAidx, DECidx], coord=[RA[(mask) & (val)], DEC[(mask) & (val)]], size=size, scale=scale)

                if comparison:
                    
                    sourceCirc = ColumnDataSource(data=dict(
                        x=ravpix,
                        y=decvpix,
                        r_g=cat['RMAG_g'][(mask) & (val)], r_f=cat['RMAG_f'][(mask) & (val)],
                        g_g=cat['GMAG_g'][(mask) & (val)], g_f=cat['GMAG_f'][(mask) & (val)],
                        z_g=cat['ZMAG_g'][(mask) & (val)], z_f=cat['ZMAG_f'][(mask) & (val)],
                        morph_g=cat['TYPE_g'][(mask) & (val)], morph_f=cat['TYPE_f'][(mask) & (val)],
                        refcat_g=cat['REF_CAT_g'][(mask) & (val)], refcat_f=cat['REF_CAT_f'][(mask) & (val)],
                        RA_g=cat['RA_g'][(mask) & (val)], RA_f=cat['RA_f'][(mask) & (val)],
                        DEC_g=cat['DEC_g'][(mask) & (val)], DEC_f=cat['DEC_f'][(mask) & (val)]
                        ))
                    
                else:
                    
                    sourceCirc = ColumnDataSource(data=dict(
                        x=ravpix,
                        y=decvpix,
                        r=cat['RMAG'][(mask) & (val)],
                        g=cat['GMAG'][(mask) & (val)],
                        z=cat['ZMAG'][(mask) & (val)],
                        morph=cat['TYPE'][(mask) & (val)],
                        refcat=cat['REF_CAT'][(mask) & (val)],
                        RA=cat['RA'][(mask) & (val)],
                        DEC=cat['DEC'][(mask) & (val)]
                        ))

                circle = p.circle('x', 'y', source=sourceCirc, size=15, fill_color=None, line_color=color, line_width=3)
                circle_i.append(circle)
                
                #circletmp = p.circle('x', 'y', source=sourceCirc, size=30, fill_color=None, line_color=color, line_width=5)
                #test_i.append(circletmp)

            lineh = Span(location=size/2, dimension='height', line_color='white', line_dash='solid', line_width=1)
            linew = Span(location=size/2, dimension='width', line_color='white', line_dash='solid', line_width=1)

            p.add_layout(lineh)
            p.add_layout(linew)

            plots.append(p)
            sources.append(circle_i)
            layers.append(layers2)
            #tests.append(test_i)
    
    checkbox = CheckboxGroup(labels=list(veto.keys()), active=list(np.arange(len(veto))))
    iterable = [elem for part in [[('_'.join(['line',str(figid),str(lineid)]),line) for lineid,line in enumerate(elem)] for figid,elem in enumerate(sources)] for elem in part]
    checkbox_code = ''.join([elem[0]+'.visible=checkbox.active.includes('+elem[0].split('_')[-1]+');' for elem in iterable])
    callback = CustomJS(args={key:value for key,value in iterable+[('checkbox',checkbox)]}, code=checkbox_code)
    checkbox.js_on_click(callback)
    
    ''' 
    radio = RadioGroup(labels=['dr9g-south', 'dr9g-south-resid'], active=0)
    iterable2 = [elem for part in [[('_'.join(['line',str(figid),str(lineid)]),line) for lineid,line in enumerate(elem)] for figid,elem in enumerate(layers)] for elem in part]
    radiogroup_code = ''.join([elem[0]+'.visible=cb_obj.active.includes('+elem[0].split('_')[-1]+');' for elem in iterable2])
    callback2 = CustomJS(args={key:value for key,value in iterable+[('radio',radio)]}, code=radiogroup_code)
    radio.js_on_change('active', callback2)
    '''
    
    radio = RadioGroup(labels=layer_list, active=3)
    iterable2 = [elem for part in [[('_'.join(['line',str(figid),str(lineid)]),line) for lineid,line in enumerate(elem)] for figid,elem in enumerate(layers)] for elem in part]
    #
    N = len(layer_list)
    text = []
    for elem in iterable2[::N]:
        for n in range(N):
            text.append('%s%s.visible=false;' %(elem[0][:-1], str(n)))
        for n in range(N):
            if n == 0: text.append('if (cb_obj.active == 0) {%s%s.visible = true;}' %(elem[0][:-1], str(0)))
            if n != 0: text.append('else if (cb_obj.active == %s) {%s%s.visible = true;}' %(str(n), elem[0][:-1], str(n)))

        radiogroup_code = ''.join(text)
    
    callback2 = CustomJS(args={key:value for key,value in iterable2+[('radio',radio)]}, code=radiogroup_code)
    radio.js_on_change('active', callback2)

    grid = gridplot(plots, ncols=cols, plot_width=256, plot_height=256, sizing_mode = None)
    
    #grid = gridplot([plots[:3]+[checkbox],plots[3:]])
    #grid = gridplot([plots+[checkbox]], plot_width=250, plot_height=250)
    
    #tab = Panel(child=p, title=layer)
    #layers.append(tab)
    
    #tabs = Tabs(tabs=[tab1, tab2])
    
    #show(row(grid,checkbox))
    #show(tabs)
    layout = column(row(radio,checkbox),grid)
    show(layout)
    
    #return iterable, checkbox_code, callback, iterable2, radiogroup_code, callback2


def plot_circle_img(coord, centeridx, veto=None, info=None, scale=0.262, scale_unit='pixscale', layer='decals-dr7', 
                    radius=None, m=4, ax=plt, isLG=None, colours=None, markers=None):
    
    from astropy.utils.data import download_file  #import file from URL
    from matplotlib.ticker import NullFormatter
    from matplotlib.patches import Ellipse
        
    RAidx = coord[0][centeridx] #centre
    DECidx = coord[1][centeridx] #centre
        
    if isLG:
        #print('Central coords in postage: RA:%.12f, DEC:%.12f, Cidx:%d, rad:%2.2g' %(RAidx, DECidx, centeridx, radius[0]*3600))
        ramin, ramax = RAidx-m*radius[0], RAidx+m*radius[0]
        decmin, decmax = DECidx-m*radius[0], DECidx+m*radius[0]
        dra = (ramax - ramin)/20
        ddec = (decmax - decmin)/20
        #postage image sizes
        boxsize = 2*m*radius[0]*3600
        size = int(round(boxsize/scale))
        rad_pix = disttopix(radius[0]*3600., scale=scale)
        major_pix = disttopix(D=radius[0]*3600, scale=scale)
        minor_pix = disttopix(D=radius[1]*3600, scale=scale)
    
        ellipse = Ellipse((size/2., size/2.), width=2*major_pix, height=2*minor_pix, angle=radius[2],
                       edgecolor='r', fc='None', lw=2, ls='-')
    else:
        #print('Central coords in postage: RA:%.12f, DEC:%.12f, Cidx:%d, rad:%2.2g' %(RAidx, DECidx, centeridx, radius*3600))
        ramin, ramax = RAidx-m*radius, RAidx+m*radius
        decmin, decmax = DECidx-m*radius, DECidx+m*radius
        dra = (ramax - ramin)/20
        ddec = (decmax - decmin)/20
        #postage image sizes
        boxsize = 2*m*radius*3600
        size = int(round(boxsize/scale))
        rad_pix = disttopix(radius*3600., scale=scale)
        angle_array = np.linspace(0, 2*np.pi, 240)
        x = size/2 - rad_pix * np.sin(angle_array)
        y = size/2 - rad_pix * np.cos(angle_array)
        #x = RAidx - radius * np.sin(angle_array)
        #y = DECidx - radius * np.cos(angle_array)
        #x_pix, y_pix = coordtopix(center=[RAidx, DECidx], coord=[x, y], size=size, scale=scale)
    
    mask = (coord[0] > ramin + dra) & (coord[0] < ramax - dra) & (coord[1] > decmin + ddec) & (coord[1] < decmax - ddec)
    #print('pixels:',size)
    scale_l = np.array([[size*5/8, size*7/8], [size*1/8, size*1/8]])

    de_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%.12f&dec=%.12f&%s=%g&layer=%s&size=%g' % (RAidx, DECidx, scale_unit, scale, layer, size)
    img = plt.imread(download_file(de_cutout_url,cache=True,show_progress=False,timeout=120))
    if ax == plt:
        fig = plt.figure(figsize=(6,6))
    else:
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    ax.imshow(img, origin='lower', aspect='equal')
    text = ax.annotate("Link", xy=(size*6.8/8,size*7.4/8), xytext=(size*6.8/8,size*7.4/8), color='yellow',
                    url='http://legacysurvey.org/viewer-dev?ra=%.12f&dec=%.12f&layer=%s&zoom=15' %(RAidx, DECidx, layer), 
                    bbox=dict(color='w', alpha=0.5, url='http://legacysurvey.org/viewer-dev?ra=%.12f&dec=%.12f&layer=%s&zoom=15' %(RAidx, DECidx, layer)), size=14)
    ax.axvline(size/2, lw=1, color='white', alpha=0.5)
    ax.axhline(size/2, lw=1, color='white', alpha=0.5)
    ax.plot(scale_l[0], scale_l[1], lw=2, color='white')
    ax.text(size*5.7/8, size*1.3/8, '%i" ' %(boxsize/4), color='yellow', size=14)
    if isLG:
        ax1 = plt.gca()
        ax1.add_patch(ellipse)
    #else:
        #ax.plot(x_pix, y_pix, 'red', lw=1)
        #ax.plot(x, y, 'red', lw=1) #uncomment to draw the fibre size
    
    if veto is not None:
        rapix = []
        decpix = []
        IDX1 = []
        j = 0
        for i in veto:
            if np.sum((veto[i]) & (mask)) < 1:
                j += 1
                continue
                
            rav = coord[0][(veto[i]) & (mask)]
            decv = coord[1][(veto[i]) & (mask)]
            #print('All coords in postage: RA:%.12f, DEC:%.12f' %(rav[1], decv[1]))
            ravpix, decvpix = coordtopix(center=[RAidx, DECidx], coord=[rav, decv], size=size, scale=scale)
            #ax.scatter(ravpix, decvpix, marker='.', s = 350, facecolors='none', edgecolors=colours[j], lw=2, label='%s' %(i))
            
            if (j > 0) & (markers is not None): ax.scatter(ravpix, decvpix, marker=markers[j], s = 40, color=colours[j], lw=2)
            else: ax.scatter(ravpix, decvpix, marker='.', s = 350, facecolors='none', edgecolors=colours[j], lw=2)
                
            for i2 in range(len(ravpix)):
                rapix.append(ravpix[i2])
                decpix.append(decvpix[i2])
                IDX1.append(list(np.where((veto[i]) & (mask))[0])[i2])
            j += 1
            
        v0 = np.zeros(len(veto[list(veto.keys())[0]]), dtype='?')
        for i,j in enumerate(list(veto.keys())):
            v0 |= veto[j]
        v0m = (~v0) & (mask)
        if np.sum(v0m) > 0:
            rav1 = coord[0][v0m]
            decv1 = coord[1][v0m]
            ravpix1, decvpix1 = coordtopix(center=[RAidx, DECidx], coord=[rav1, decv1], size=size, scale=scale)
            #ax.scatter(ravpix1, decvpix1, marker='.', s = 300, facecolors='none', edgecolors='red', lw=2, label='other')
            ax.scatter(ravpix1, decvpix1, marker='.', s = 300, facecolors='none', edgecolors='red', lw=2)
            for i1 in range(len(ravpix1)):
                rapix.append(ravpix1[i1])
                decpix.append(decvpix1[i1])
                IDX1.append(list(np.where(v0m)[0])[i1])
        
        if info is not None:
            j2 = 9*size/10
            for k in range(len(rapix)):
                ax.text(rapix[k]+rad_pix*0.20, decpix[k]-rad_pix*0.20, '%s' %(k), color='white', fontsize=12)
                txt = []
                for l in info.keys():
                    val = info[l][IDX1[k]]
                    if isinstance(val, (float, np.float32, np.float64)):
                        txti = '%s:%2.4g' %(l,val)
                    elif isinstance(val, str):
                        txti = '%s:%s' %(l,val)
                    elif isinstance(val, int):
                        txti = '%s:%i' %(l,val)
                    txt.append(txti)
                if IDX1[k] == centeridx:
                    colorLab = 'white'
                else:
                    colorLab = 'yellow'
                ax.text(size/16,j2, '%i) %s' %(k, ','.join(txt)), fontsize=8,color=colorLab, alpha=0.8)
                j2 -= size/24
                
        #ax.legend(loc = 'upper right')
    else:
        rav = coord[0][mask]
        decv = coord[1][mask]
        rapix, decpix = coordtopix(center=[RAidx, DECidx], coord=[rav, decv], size=size, scale=scale)
        ax.scatter(rapix, decpix, marker='.', s = 300, facecolors='none', edgecolors='lime', lw=2)
    
    if ax == plt:
        return fig
    

def postages_circle(coord, centeridx, veto=None, info=None, scale=0.262, scale_unit='pixscale', layer='decals-dr7', 
                    radius=None, m=4, grid=None, savefile=None, layer2=None, layer2Mode='merge', isLG=False, 
                    title=None, markers=True, colorkey=True):
    '''
    Create a postage image (or a table of images) from selected object(s).
    
    coord::class:`2D-array`
        RA, DEC coordinates of catalogue/dataframe of interest.
    centeridx::class:`array or int`
        Index(es) of the object(s) that will be at the centre of the postage(s). The index have to follow the catalogue/dataframe indexes.
    veto::class:`dictionary-boolean-array`
        Dictionary-array containing boolean-arrays that will be shown in the postages as labels. These need to have same lenght as coord and same indexes.
    info::class:`dictionary-boolean-array`    
        
        '''
    
    import matplotlib.gridspec as gridspec
    import random
    import time
    import progressbar
    
    veto_colours = ['lime', 'royalblue', 'purple', 'orange', 'yellow']
    if markers: mark = ['+', 'x', '*']
    else: mark = None

    if grid is not None:
        
        if not isinstance(centeridx, np.ndarray):
            raise ValueError('If using grid use a proper list of index at centeridx...')
            
        rows, cols = grid[0], grid[1]
        if (layer2 is not None) & (layer2Mode == 'merge'):
            figsize = (4.5*cols, 4.5*rows*2)
            gs = gridspec.GridSpec(rows*2, cols)
        else:
            figsize = (4.5*cols, 4.5*rows)
            gs = gridspec.GridSpec(rows, cols)
            
        gs.update(wspace=0.001, hspace=0.06)
        idx_list = random.sample(list(centeridx), rows*cols)

        fig = plt.figure(num=1, figsize=figsize)
        ax = []
        
        widgets = ['\x1b[32mProgress...\x1b[39m', progressbar.Percentage(),progressbar.Bar(markers='\x1b[32m$\x1b[39m')]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=rows*cols).start()
        #bar = progressbar.ProgressBar(widgets=[progressbar.SimpleProgress()],max_value=rows*cols*10,).start()
        
        for i, j in enumerate(idx_list):
            
            if isinstance(radius, np.ndarray):
                radius2 = radius[j]
                #print(j, radius2*3600)
            else:
                radius2 = radius
                
            if (layer2 is not None) & (layer2Mode == 'merge'):
                row = (i // cols)*2
            else:
                row = (i // cols)
            col = i % cols

            ax.append(fig.add_subplot(gs[row, col]))
            plot_circle_img(coord=coord, centeridx=j, veto=veto, info=info, scale=scale, 
                scale_unit=scale_unit, layer=layer, radius=radius2, m=m, ax=ax[-1], isLG=isLG, colours=veto_colours, markers=mark)
            
            if (layer2 is not None) & (layer2Mode == 'merge'):
                ax.append(fig.add_subplot(gs[row+1, col]))
                plot_circle_img(coord=coord, centeridx=j, veto=veto, info=info, scale=scale, 
                        scale_unit=scale_unit, layer=layer2, radius=radius2, m=m, ax=ax[-1], isLG=isLG, colours=veto_colours, markers=mark)
                    
            if (layer2 is not None) & (layer2Mode == 'separate'):
                fig2 = plt.figure(num=2, figsize=figsize)
                ax2 = []
                ax2.append(fig2.add_subplot(gs[row, col]))
                plot_circle_img(coord=coord, centeridx=j, veto=veto, info=info, scale=scale, 
                        scale_unit=scale_unit, layer=layer2, radius=radius2, m=m, ax=ax2[-1], isLG=isLG, colours=veto_colours, markers=mark)
                
            time.sleep(0.1)
            bar.update(i + 1)
            
    else:
        if isinstance(centeridx, np.ndarray):
            raise ValueError('If Not using grid do not use a list of index at centeridx...')
            
        if (isinstance(radius, np.ndarray)) & (~isLG):
            raise ValueError('If Not using grid do not use a list of radius...')
        
        fig = plot_circle_img(coord=coord, centeridx=centeridx, veto=veto, info=info, scale=scale, 
                scale_unit=scale_unit, layer=layer, radius=radius, m=m, ax=plt, isLG=isLG, colours=veto_colours, markers=mark)
    
    if title is not None:
        fig.suptitle(r'%s' %(title), y=0.89, size=18)
    
    if savefile is not None:
        if (layer2 is not None) & (layer2Mode == 'separate'):
            fig.canvas.print_figure(savefile +'.svg', bbox_inches = 'tight', pad_inches = 0)
            fig2.canvas.print_figure(savefile + '_%s' %(layer2[-5:]) + '.svg', bbox_inches = 'tight', pad_inches = 0)
        elif (layer2 is not None) & (layer2Mode == 'merge'):
            fig.canvas.print_figure(savefile + '_%s' %(layer2[-5:]) + '.svg', bbox_inches = 'tight', pad_inches = 0)
        else:
            fig.canvas.print_figure(savefile +'.svg', bbox_inches = 'tight', pad_inches = 0)
            
    bar.finish()
    
    if colorkey:
        print('Colour key:')
        for i, key in enumerate(veto.keys()):
            print('\t %s --> %s' %(key, veto_colours[i]))
        print('\t other --> red')
        
    