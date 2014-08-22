# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:46:40 2014

@author: Parke
"""

import pdb
import numpy as np
import my_numpy as mynp
import matplotlib.pyplot as plt
import plotutils as pu
from math import erf
from scipy.special import gammainc

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
    if type(scalefunc) is float:
        exponent = scalefunc
        scalefunc = lambda x: x**exponent
    if bins == None: bins = np.sqrt(len(x)/25)
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
    backsub = (tback != None)
    weighted = (eps != None)
    checkrng = lambda r,x: [np.nanmin(x), np.nanmax(x)] if r == None else r
    trange, wrange = map(checkrng, [trange, wrange], [t,w])
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
    
    if dN == None: wedges, cpsw, cpswerr = map(np.array, [wedges, cpsw, cpswerr])
    return tedges, wedges, cpsw, cpswerr

def spectrum(w, wback=None, eps=None, epsback=None, area_ratio=None, 
             wbins=None, dN=None, wrange=None):
    """Computes a spectrum (weighted counts per wavelength) from a list of
    photons.
    
    w = event wavelengths
    y = event cross dispersion location (option, required with ysignal and yback)
    eps = event weights
    
    User can supply one of:
    wbins = number (int), width (float), or edges (iterable) of bins
    dN = target number of signal(!) counts per bin (actual number will
         vary due to groups of idnetically-valued points)
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
    if not (wbins != None or dN): wbins = np.sqrt(len(w))
    if wrange == None: wrange = [np.nanmin(w), np.nanmax(w)]
    if wback != None and area_ratio == None: raise ValueError(needaratio)
    weighted = (eps != None)
    
    #bin counts according to wedges, Nbins, or dN
    if wbins != None: 
        wedges = __bins2edges(wrange, wbins)
        signal = np.histogram(w, bins=wedges, range=wrange, weights=eps)[0]
        varsignal = np.histogram(w, bins=wedges, range=wrange, weights=eps**2)[0] if weighted else signal
    if dN:
        w = np.concatenate([[wrange[0]], w, [wrange[1]]])
        signal, wedges = mynp.chunkogram(w, dN, weights=eps, unsorted=True)
        varsignal = mynp.chunkogram(w, dN, weights=eps, unsorted=True)[0] if weighted else signal
    
    if wback != None:
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
    if tbins != None and dN:
        print 'Only specify tbins (time bins) or dN (count bins), but not both.'
        return
    if not tbins != None and not dN:
        dN = 100
        print 'Using default {} point bins.'.format(dN)
    Ncurves = len(groups) if groups != None else len(bands)
    
    if bands == None: bands = [[np.nanmin(w), np.nanmax(w)]]
    bands = np.array(bands)
    order = np.argsort(bands, 0)[:,0]
    bands = bands[order]
    wedges = np.ravel(bands)
    if any(wedges[1:] < wedges[:-1]):
        print ('Wavelength bands cannot overlap. For overlapping bands, you '
               'must call the function multiple times.')
        return
    
    if trange == None: trange = [np.nanmin(t), np.nanmax(t)]
    
    weighted = (eps != None)
    backsub = (tback != None)
    if backsub and (area_ratio == None): raise ValueError(needaratio)
        
##### BIN COUNTS ##############################################################
    pts = np.array([t,w]) if eps == None else np.array([t,w,eps])
    if backsub: ptsback = np.array([tback,wback])
    if backsub and weighted: ptsback = np.vstack([ptsback, epsback])
    
    # divide up points into wavelength bands (discard between pts via ::2)
    def divvy(pts):
        templist = mynp.divvy(pts, wedges, keyrow=1)[::2]
        original_order = np.argsort(order)
        if groups != None: #combine lines as specified by groups
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
    tedges, countssignal, varsignal= list(), list(), list()
    countsback, varback = [list(), list()] if backsub else [None, None]

    #all tedges are the same if using tbins bins
    tedges = [__bins2edges(trange, tbins)]*Ncurves if tbins != None else [None]*Ncurves
    
    #first bin the signal counts
    for i,pts in enumerate(ptslist):
        t = pts[0,:]
        e = pts[-1,:] if weighted else None
        if tbins != None:
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
    if back != None:
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
    ys = list(ysignal)
    if yback != None: yb = [list(b) for b in yback]
    ybands = yb + [ys] if yback != None else [ys]
    ybands.sort(key=lambda y: y[0])
    edges = reduce(lambda x,y:x+y, ybands)
    
    #record signal and background positions in that list
    isignal = edges.index(ys[0])
    try:
        iback = np.array([edges.index(yr[0]) for yr in yb])
    except ValueError:
        iback = None
    
    #check for bad input
    edges = np.array(edges)
    if any(edges[1:] < edges[:-1]):
        raise ValueError('Signal and/or background ribbons overlap. This '
                         'does not seem wise.')
    
    #divide up the counts
    div_cnts = mynp.divvy(cnts, edges, keyrow=yrow) 
    
    if yb != None:
        #compute ratio of signal to background area
        area_signal = ys[1] - ys[0]
        area_back = sum([yr[1] - yr[0] for yr in yb])
        area_ratio = float(area_signal)/area_back
    
    signal = div_cnts[isignal]
    back = np.hstack([div_cnts[i] for i in iback])
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
    Nfull = (rng[1] - rng[0]) // d
    temp = np.zeros(Nfull+1)
    temp[0] = rng[0]
    temp[1:] = d*np.ones(Nfull)
    bins = np.cumsum(temp)
    if (rng[1] - rng[0]) % d > d*0.5:
        return np.append(bins,rng[1])
    else:
        return bins
        
def identify_continuum(wbins, y, err, function_generator, maxsig=2.0, 
                       emission_weight=1.0, maxcull=0.99, plotsteps=False):
    if plotsteps: plt.ioff()
    if len(wbins) != len(y):
        raise ValueError('The shape of wbins must be [len(y), 2]. These '
                         'represent the edges of the wavelength bins over which '
                         'photons were counted (or flux was integrated).')
    wbins, y, err = map(np.array, [wbins, y, err])
    Npts = len(y)
    
    if plotsteps:
        wave = (wbins[:,0] + wbins[:,1])/2.0
        waveold, yold = wave, y
        
    while True:
        #fit to the retained data
        f = function_generator(wbins, y, err)
        
        #count the runs
        expected = f(wbins)
        posneg = (y > expected)
        run_edges = ((posneg[1:] - posneg[:-1]) !=0)
        Nruns = np.sum(run_edges) + 1
        
        if plotsteps:
            plt.plot(waveold, yold)
            plt.plot(wave,expected,'k')
            plt.plot(wave,y,'g.')
        
        #compute the PTE for the runs test
        N = len(y)
        Npos = np.sum(posneg)
        Nneg = N - Npos
        mu = 2*Npos*Nneg/N + 1
        var = (mu-1)*(mu-2)/(N-1)
        sigruns = abs(Nruns - mu)/np.sqrt(var)
        
        #if the fit passes the runs test, then return the good wavelengths
        if sigruns < maxsig:
            non_repeats = (wbins[:-1,1] != wbins[1:,0])
            w0, w1 = wbins[1:,0][non_repeats], wbins[:-1,1][non_repeats]
            w0, w1 = np.insert(w0, 0, wbins[0,0]), np.append(w1, wbins[-1,1])
            return np.array([w0,w1]).T
        else:
            #compute the chi2 PTE for each run
            iedges = np.concatenate([[0], np.nonzero(run_edges)[0]+1, [len(run_edges)+1]]) #the slice index
            chiterms = ((y - expected)/err)**2
            chisum = np.cumsum(chiterms)
            chiedges = np.insert(chisum[iedges[1:]-1], 0, 0.0)
            chis =  chiedges[1:] - chiedges[:-1]
            DOFs = (iedges[1:] - iedges[:-1])
            sigs = abs(chis - DOFs)/np.sqrt(2*DOFs)
            if emission_weight != 1.0:
                if posneg[0] > 0: sigs[::2] = sigs[::2]*emission_weight
                else: sigs[1::2] = sigs[1::2]*emission_weight
            
#            PTEs = 1.0 - gammainc(DOFs/2.0, chis/2.0)
            
            #mask out the runs with PTEs too low to be expected given the
            #number of runs (e.g. if there are 10 runs, roughly one run should
            #have a PTE < 10%). Always mask out at least one or we could enter
            #an infinite loop.
#            good = (PTEs > 1.0/Nruns/1000.0)
#            good = (sigs < maxsig)
#            if np.sum(good) == Nruns: #if none would be masked out
            good = np.ones(len(y), bool)            
            good[np.argmax(sigs)] = False #mask out the run with the smallest PTE
            keep = np.concatenate([[g]*d for g,d in zip(good, DOFs)]) #make boolean vector
            
            if plotsteps:
                trash = np.logical_not(keep)
                plt.plot(wave[trash], y[trash], 'r.') 
                ax = plt.gca()
                plt.text(0.8, 0.9, '{}'.format(sigruns), transform=ax.transAxes)
                plt.show()
                wave = wave[keep]
            wbins, y, err = wbins[keep], y[keep], err[keep] 
        
        if float(len(y))/Npts < (1.0 - maxcull):
            raise ValueError('More than maxcull={}% of the data has been '
                             'removed, but the remaining data and associated '
                             'fit is still not passing the Runs Test. Consider '
                             'checking that the fits are good, relaxing '
                             '(increasing) the maxsig condition for passing '
                             'the Runs Test, or increasing maxcull to allow '
                             'more data to be masked out.')