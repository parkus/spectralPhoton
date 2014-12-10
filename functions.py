# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:46:40 2014

@author: Parke
"""

import numpy as np
import my_numpy as mynp
import matplotlib.pyplot as plt
import plotutils as pu

#some reused error messages
needaratio = ('If background counts are supplied, the ratio of the signal '
              'extraction area to background extraction area must also be '
              'supplied.')

def image(x, y, eps=None, bins=None, scalefunc=None, **kw):
    """Displays an image of the counts.
    
    bins=None results in bins=sqrt(len(x)/25) such that each bin would have
    25 counts in a perfectly flat image (S/N = 5)
    otherwise bins behaves as in numpy.histrogram2d
    
    scalefunc is a function to scale the image, such as
    scalefunc = lambda x: np.log10(x)
    for log scaling. It can also be a float, in which case pixel values will
    be exponentiated by the float.
    """
    x, y, eps = map(__asdblary, [x, y, eps])
    
    if type(scalefunc) is float:
        exponent = scalefunc
        scalefunc = lambda x: x**exponent
    if bins is None: bins = np.sqrt(len(x)/25)
    h = np.histogram2d(x, y, weights=eps, bins=bins)
    img = scalefunc(h[0]) if scalefunc else h[0]
    img = np.transpose(img)
    x, y = mynp.midpts(h[1]), mynp.midpts(h[2])
    p = pu.pcolor_reg(x, y, img, **kw)
    plt.xlim(np.nanmin(h[1]), np.nanmax(h[1]))
    plt.ylim(np.nanmin(h[2]), np.nanmax(h[2]))
    del(h,img)
    return p

def spectrum_frames(t, w, tback=None, wback=None, eps=None, epsback=None, 
                    area_ratio=None, tbins=None, dN=None, wbins=None, trange=None, 
                    wrange=None):
    """Generates spectra at set time intervals via spectrum() in
    counts/time/wavelength (unlike spectrum(), which returns counts/wavlength).
    """  
    t, w, tback, wback, eps, epsback = map(__asdblary, [t, w, tback, wback,
                                                        eps, epsback])
    
    backsub = (tback is not None)
    weighted = (eps is not None)
    checkrng = lambda r,x,b: __range(x,b) if r is None else r
    trange, wrange = map(checkrng, [trange, wrange], [t,w], [tbins, wbins])
    tedges = __bins2edges(trange, tbins)
    
    #TODO:use a 2d histogram when not using dN for faster speed
    pts = [t,w,eps] if weighted else [t,w]
    div = mynp.divvy(np.array(pts), tedges)
    if backsub: 
        ptsback = [tback, wback, epsback] if weighted else [tback, wback]
        divback = mynp.divvy(np.array(ptsback), tedges)
    
    wedges, cpsw, cpswerr = [], [], []
    for i in range(len(div)):
        w = div[i][1,:]
        eps = div[i][2,:] if weighted else None
        wback = divback[i][1,:] if backsub else None
        epsback = divback[i][2,:] if (backsub and weighted) else None
        we, cpw, err = spectrum(w, wback=wback, eps=eps, epsback=epsback,
                                area_ratio=area_ratio, wbins=wbins, dN=dN, wrange=wrange)
        dt = tedges[i+1] - tedges[i]
        wedges.append(we)
        cpsw.append(cpw/dt)
        cpswerr.append(err/dt)
    
    if dN is None: wedges, cpsw, cpswerr = map(np.array, [wedges, cpsw, cpswerr])
    return tedges, wedges, cpsw, cpswerr

def spectrum(w, wback=None, eps=None, epsback=None, area_ratio=None, 
             wbins=None, dN=None, wrange=None):
    #TODO: using DN is not providing correct results! check with G230L spectrum
    #from ak sco and see what is going on
    """Computes a spectrum (weighted counts per wavelength) from a list of
    photons.
    
    w = event wavelengths
    y = event cross dispersion location (option, required with ysignal and yback)
    eps = event weights
    
    User can supply one of:
    wbins = number (int), width (float), or edges (iterable) of bins
    dN = target number of signal(!) counts per bin (actual number will
         vary due to groups of identically-valued points)
    If none are supplied, the function uses Nbins=sqrt(len(w))
    
    yback = [[min0,max0], [min1,max1], ...] yvalues to be considered background
            counts
    
    wrange = [min,max] wavelengths to consider -- saves the computation time of
            determining min and max from the w vector
    
    Returns [wedges, cpw, cpw_err]: the edges of the wavelength bins, counts
    per wavelength for each bin, and the Poisson error
    
    epsback contains area ratio information, can be scalar
    """
    #groom input
    w, wback, eps, epsback = map(__asdblary, [w, wback, eps, epsback])
    if not (wbins is not None or dN): wbins = np.sqrt(len(w))
    if wrange is None: wrange = __range(w, wbins)
    if wback is not None and area_ratio is None: raise ValueError(needaratio)
    weighted = (eps is not None)
    
    #bin counts according to wedges, Nbins, or dN
    if wbins is not None: 
        wedges = __bins2edges(wrange, wbins)
        signal = np.histogram(w, bins=wedges, range=wrange, weights=eps)[0]
        varsignal = np.histogram(w, bins=wedges, range=wrange, weights=eps**2)[0] if weighted else signal
    if dN:
        w = np.concatenate([[wrange[0]], w, [wrange[1]]])
        signal, wedges = mynp.chunkogram(w, dN, weights=eps, unsorted=True)
        varsignal = mynp.chunkogram(w, dN, weights=eps, unsorted=True)[0] if weighted else signal
    
    if wback is not None:
        back = np.histogram(wback, wedges, weights=epsback)[0]
        varback = np.histogram(wback, wedges, weights=epsback**2)[0] if weighted else back
    else: back, varback = None, None
        
    #compute count density
    minvar = np.nanmin(eps)**2 if weighted else 1.0
    cpw, cpw_err = __count_density(wedges, signal, varsignal, back, varback,
                                  area_ratio, minvar)
    
    return wedges, cpw, cpw_err
    
def spectral_curves(t, w, tback=None, wback=None, bands=None, dN=None, tbins=None, 
                    eps=None, epsback=None, area_ratio=None, trange=None, groups=None):
    """Makes lightcurves from a photon list in the specified spectral bands.
    
    t = event times in s
    w = event wavelengths
    y = event distance from spectrum midline (can be set to None and is ignored
        unless yback and ysignal are both supplied)
    eps = event weight (optional)
    bands = [[wmin0,wmax0], [wmin1, wmax1], ...] limits of spectral bands to
            create curves from. If None, full wavelength range is used.
    dN = number of points per time bin
    tbins = number (int), width (float), or edges (iterable) of time bins
    trange = start and end times of the exposure (optional, assumed to be the
    time of the first and last photon if not supplied)
    groups = 2D list/array of integers specifying how lines should be combined, e.g. 
    [[0,3],[1,2,4]] would produce two light curves, the first combining counts
    from the first and fourth bands, and the second from the second, third, 
    and fifth.
    
    dN and tbins cannot both be used - the bins must either be constructed from
    set numbers of events or from set time steps 
    
    returns [tstart, tend, cpt, cpt_err]: the start time of the bins,
    end time of the bins, the (weighted) count rate and the Poisson error
    notes:
    if ysignal and yback regions overlap, they will be made not to
    be careful to get the nested lists right -- i.e. for bands,yback, and tgood
    """
    #groom input
    t, w, tback, wback, eps, epsback = map(__asdblary, [t, w, tback, wback,
                                                        eps, epsback])
    if tbins is not None and dN:
        print 'Only specify tbins (time bins) or dN (count bins), but not both.'
        return
    if not tbins is not None and not dN:
        dN = 100
        print 'Using default {} point bins.'.format(dN)
    Ncurves = len(groups) if groups is not None else len(bands)
    
    if bands is None: bands = [[np.nanmin(w), np.nanmax(w)]]
    bands = np.array(bands)
    order = np.argsort(bands, 0)[:,0]
    bands = bands[order]
    wedges = np.ravel(bands)
    if any(wedges[1:] < wedges[:-1]):
        print ('Wavelength bands cannot overlap. For overlapping bands, you '
               'must call the function multiple times.')
        return
    
    if trange is None: trange = __range(t,tbins)
    
    weighted = (eps is not None)
    backsub = (tback is not None)
    if backsub and (area_ratio is None): raise ValueError(needaratio)
        
##### BIN COUNTS ##############################################################
    pts = np.array([t,w]) if eps is None else np.array([t,w,eps])
    if backsub: ptsback = np.array([tback,wback])
    if backsub and weighted: ptsback = np.vstack([ptsback, epsback])
    
    # divide up points into wavelength bands (discard between pts via ::2)
    def divvy(pts):
        templist = mynp.divvy(pts, wedges, keyrow=1)[::2]
        original_order = np.argsort(order)
        if groups is not None: #combine lines as specified by groups
            def stack(c):
                temp = np.hstack([templist[i] for i in original_order[c]])
                return temp[:, np.argsort(temp[0])]
            ptslist = map(stack,groups)
        else: #sort back into the orignal order
            ptslist = [templist[i] for i in original_order]
        return ptslist
    ptslist = divvy(pts)
    del pts
    if backsub: 
        ptsbacklist = divvy(ptsback)
        del ptsback
    
    #prep lists
    countssignal, varsignal= list(), list()
    countsback, varback = [list(), list()] if backsub else [None, None]

    #all tedges are the same if using tbins bins
    tedges = [__bins2edges(trange, tbins)]*Ncurves if tbins is not None else [None]*Ncurves
    
    #first bin the signal counts
    for i,pts in enumerate(ptslist):
        t = pts[0,:]
        e = pts[-1,:] if weighted else None
        if tbins is not None:
            sums = np.histogram(t, bins=tedges[0], weights=e)[0]
            quads = np.histogram(t, bins=tedges[0], weights=e**2)[0] if weighted else sums
        if dN:
            #if there are less than dN counts, deal with it
            if len(t) < dN:
                tedges[i] = np.array(trange)
                sums = [np.sum(e)] if weighted else [np.array([float(len(t))])]
                quads = [np.sum(e**2)] if weighted else [np.array([float(len(t))])]
            else:
                #tack on pretend photons to deal with exposure start and end
                dt0, dt1 = (t[0] - trange[0]), (trange[1] - t[-1])
                t0, t1 = (trange[0] - dt0), (trange[1] + dt1)
                text = np.insert(t, [0,len(t)], [t0,t1])
                te = mynp.chunk_edges(text, dN)
                
                #append the last odd chunk if necessary
                if te[-1] < t[-1]:
                    Nchunks = len(te)
                    te = np.append(te, t[-1])
                    sums, quads = np.zeros(Nchunks), np.zeros(Nchunks)
                    oddcnts = (t > te[-2])
                    sums[-1] = np.sum(e[oddcnts]) if weighted else np.sum(oddcnts)
                    quads[-1] = np.sum(e[oddcnts]**2) if weighted else np.sum(oddcnts)
                    regchunks = np.arange(Nchunks-1)
                else:
                    Nchunks = len(te)-1
                    sums, quads = np.zeros(Nchunks), np.zeros(Nchunks)
                    regchunks = np.arange(Nchunks)
                    
                #count up the signal photons
                tedges[i] = te
                sums[regchunks] = mynp.chunk_sum(e, dN) if weighted else float(dN)
                quads[regchunks] = mynp.chunk_sum(e**2, dN) if weighted else float(dN)
            
        countssignal.append(sums)
        varsignal.append(quads)
        
    #now the background counts
    if backsub:
        for pts, te in zip(ptsbacklist, tedges):
            t = pts[0,:]
            e = pts[-1,:] if weighted else None
            sums = np.histogram(t, te, weights=e)[0]
            quads = np.histogram(t, te, weights=e**2)[0] if weighted else sums
            countsback.append(sums)
            varback.append(quads)
    else:
        countsback, varback = [[None]*Ncurves]*2
        
    #compute the count rates
    cps, cps_err = [], []
    minvar = np.min(eps[eps > 0.0])**2 if weighted else 1.0
    for i in range(Ncurves):
        result = __count_density(tedges[i], countssignal[i], varsignal[i], 
                                 countsback[i], varback[i], area_ratio, minvar)
        cps.append(result[0])
        cps_err.append(result[1])
    
    return tedges,cps,cps_err
    
def __count_density(xvec, signal, varsignal, back, varback, area_ratio, minvar):
    deltas = xvec[1:] - xvec[:-1]
    var = np.copy(varsignal) #if identical to signal, it was not copied earlier
    var[var == 0] = minvar
    if back is not None:
        cps = (signal - back*area_ratio)/deltas
        err = np.sqrt(var + varback*area_ratio)/deltas
    else:
        cps = signal/deltas
        err = np.sqrt(var)/deltas
    return cps, err
    
def divvy_counts(cnts, ysignal, yback=None, yrow=0):
    """Divvies up the signal and backgroudn counts in the cnts array according
    to their y-values (y-values assumed to be in first row of array).
    
    cnts = array of counts with each row a different dimension (time,
           wavelength, etc)    
    ysignal = [ymin,ymax] limits of signal region
    yback = [[ymin0,ymax0], [ymin1,ymax1], ...] limits of background regions
    eps = count weights
    """
    #join the edges into one list for use with mnp.divvy
    ys = np.array(ysignal) if yback is None else np.append(ysignal, yback) 
    args = np.argsort(ys)
    edges = ys[args]
    
    #record signal and background positions in that list
    isignal, iback = args[0], args[2::2]
    
    #check for bad input
    if any(edges[1:] < edges[:-1]):
        raise ValueError('Signal and/or background ribbons overlap. This '
                         'does not seem wise.')
    
    #divide up the counts
    div_cnts = mynp.divvy(cnts, edges, keyrow=yrow) 
    signal = div_cnts[isignal]
    
    if yback is None:
        return signal
    else:
        back = np.hstack([div_cnts[i] for i in iback])
        #compute ratio of signal to background area
        area_signal = ys[1] - ys[0]
        area_back = np.sum(ys[3::2] - ys[2::2])
        area_ratio = float(area_signal)/area_back
        return signal, back, area_ratio

def __bins2edges(rng, d):
    if isinstance(d, float):
        edges = __oddbin(rng, d)
    elif isinstance(d, int):
        edges = np.linspace(rng[0], rng[1], d+1)
    elif hasattr(d, '__iter__'):
        edges = d
    else:
        raise ValueError('Input bin must be either an integer, float, or iterable.')
    return edges
    
def __oddbin(rng, d):
    bins = np.arange(rng[0], rng[1], d)
    if (rng[1] - bins[-1]) > d*0.5:
        return np.append(bins,rng[1])
    else:
        return bins
        
def __range(vals, bins):
    if hasattr(bins, '__iter__'):
        return [bins[0], bins[-1]]
    else:
        return [np.min(vals), np.max(vals)]
        
def __asdblary(ary):
    return np.asarray(ary, 'f8') if ary is not None else None