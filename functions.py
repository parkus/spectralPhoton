# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:46:40 2014

@author: Parke
"""

import numpy as np
import mypy.my_numpy as mynp
import matplotlib.pyplot as plt
import mypy.plotutils as pu
from math import log10

#some reused error messages
needaratio = ('If background counts are supplied, the ratio of the signal '
              'extraction area to background extraction area must also be '
              'supplied.')

def image(x, y, eps=None, bins=None, scalefunc=None, **kw):
    """Displays an image of the counts.

    bins=None results in bins=sqrt(len(x)/25) such that each bin would have
    25 counts in a perfectly flat image (S/N = 5)
    otherwise bins behaves as in numpy.histrogram2d

    Parameters
    ----------
    x : 1d array-like
        x coordinates of the counts
    y : 1d array-like
        y coordinates of the counts
    eps : 1d array-like, optional
        weights of the counts
    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:
            - If int, the number of bins for the two dimensions (nx=ny=bins).
            - If [int, int], the number of bins in each dimension (nx, ny = bins).
            - If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
            - If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
    scalefunc : {function|float|'auto'}
        Scale function to apply to the histogrammed count values:
            - If function, the count values are passed as an argument and the
                result is used to create the image. E.g.:
                scalefunc = lambda x: np.log10(x)
            - If float, count values will be exponetiated by float before
                creating image.
            - If 'auto', the image will be scaled by the histogram, such
                that prevalent pixel values are highlighted.
        Keywords to be passed to the matplotlib.pyplot.imshow function.

    Returns
    -------
    p : object
        Handle for the image plot.
    """
    #histogram
    x, y, eps = map(__asdblary, [x, y, eps])
    if bins is None: bins = np.sqrt(len(x)/25)
    h = np.histogram2d(x, y, weights=eps, bins=bins)

    #scale, if desired
    if scalefunc == 'auto':
        bins = np.arange(np.max(h[0]) + 2)
        h1 = np.histogram(h[0], bins)[0]
        sums = np.append([0.0], np.cumsum(h1[1:]))
        scalepts = sums.astype(float)/sums[-1]
        def scalefunc(arr):
            i = arr.astype(int)
            vals = scalepts[i.flatten()]
            return np.reshape(vals, arr.shape)
    if type(scalefunc) is float:
        exponent = scalefunc
        scalefunc = lambda x: x**exponent
    img = scalefunc(h[0]) if scalefunc else h[0]
    img = np.transpose(img)

    #create image
    x, y = mynp.midpts(h[1]), mynp.midpts(h[2])
    p = pu.pcolor_reg(x, y, img, **kw)
    plt.xlim(np.nanmin(h[1]), np.nanmax(h[1]))
    plt.ylim(np.nanmin(h[2]), np.nanmax(h[2]))

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

def smooth_curve(t, w, eps, n, bands=None, trange=None):
    """
    #TODO: write docstring
    """
    # GROOM INPUT
    #------------
    t, w, eps = map(__asdblary, [t, w, eps])

    assert np.all(t[1:] > t[:-1])

    if bands is None: bands = [[np.nanmin(w), np.nanmax(w)]]
    if trange is None:
        trange = t[[0,-1]]
        t, w, eps = t[1:-1], w[1:-1], eps[1:-1]

    #FIXME: make separate function for the stuff in common with spectral_curves
    bands = np.array(bands)
    order = np.argsort(bands, 0)[:,0]
    bands = bands[order]
    wedges = np.ravel(bands)
    if any(wedges[1:] < wedges[:-1]):
        raise ValueError('Wavelength bands cannot overlap.')

    # PARSE IN-BAND COUNTS
    #---------------------
    ## determine where photons fall in order of wavelength edges
    i = np.searchsorted(wedges, w)

    ## keep only those that fall within a band (even number in sort order)
    keep = (i % 2 == 1)
    t = t[keep]
    eps = eps[keep]

    # SMOOTH
    #-------
    cnt = mynp.smooth(eps, n, safe=True) * n
    cnterr = np.sqrt(mynp.smooth(eps**2, n, safe=True) * n)

    # GET TIME BINS
    #--------------
    # get midpoints between counts
    temp = np.insert(t, [0, len(t)], trange)
    t2 = mynp.smooth(temp, 2, safe=True)
    t0 = t2[:-n]
    t1 = t2[n:]
    dt = t1 - t0

    return t0, t1, cnt/dt, cnterr/dt


def smooth_spec(w, eps, n, waverange=None):
    """
    #TODO: write docstring

    Note that the output is not divided by time.
    """
    # GROOM INPUT
    #------------
    w, eps = map(__asdblary, [w, eps])

    isort = np.argsort(w)
    w, eps = w[isort], eps[isort]
    if waverange is None:
        waverange = w[[0, -1]]

    # SMOOTH
    #-------
    cnt = mynp.smooth(eps, n, safe=True) * n
    cnterr = np.sqrt(mynp.smooth(eps**2, n, safe=True) * n)

    # GET WAVE BINS
    #--------------
    # get midpoints between counts
    temp = np.insert(w, [0, len(w)], waverange)
    w2 = mynp.smooth(temp, 2, safe=True)
    w0 = w2[:-n]
    w1 = w2[n:]
    dw = w1 - w0

    return w0, w1, cnt/dw, cnterr/dw


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
    edges, isignal, iback, area_ratio = __form_edges(ysignal, yback, cnts, yrow)

    # since divvy excludes stuff outside of bins, we need to decrement indices
    iback -= 1
    isignal -= 1

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

        return signal, back, area_ratio

def squish(counts, ysignal, yback=None, yrow=0, weightrows=None):
    """Extracts counts from the signal and background regions and returns them
    as a single array.

    Background count weights are multiplied by -ar, where ar is the ratio of
    the signal area to the background area. This way, the weights can be used
    with histogram for rapid binning.

    Parameters
    ---------
    counts : 2D array-like
        The counts, with each row a different dimension (e.g. time, wavelength).
    ysignal : 1D array-like, length 2, or list of
        [ymin,ymax] limits of signal region
    yback : 2D array-like, size Nx2, optional, or list of
        [[ymin0,ymax0], [ymin1,ymax1], ...] limits of background regions
    yrow : int or list, optional
        The row of counts that represents the cross-dispersion information.
    weightrows : 1D array-like, optional
        Rows of the counts array that represent count weights. If none, the
        function will append a row with unity weights for all signal counts
        and -ar weights for all background counts. Otherwise, the weights in
        the specified rows will be multiplied by these values.

    Returns
    -------
    photons : 2D array
        The counts array with counts not in a signal or background region removed
        and weights modified or appended as appropriate.
    """
    ysignal, yback = map(np.asarray, [ysignal, yback])
    if ysignal.ndim == 2 or yback.ndim == 3:
        if ysignal.ndim < 2 or yback.ndim < 3 or not (len(yrow) == len(ysignal) == len(yback)):
            raise ValueError('If orders are supplied, then ysignal and yback regions must be supplied for each order.')
        else:
            cntsList = []
            for ys, yb, yr in zip(ysignal, yback, yrow):
                cntsList.append(squish(counts, ys, yb, yr, weightrows))
            return np.hstack(cntsList)

    cnts = np.copy(counts)

    #join the edges into one list
    edges, isignal, iback, area_ratio = __form_edges(ysignal, yback, cnts, yrow)

    #check for bad input
    if any(edges[1:] < edges[:-1]):
        raise ValueError('Signal and/or background ribbons overlap. This '
                         'does not seem wise.')

    #determine which band counts are in
    ii = np.searchsorted(edges, cnts[yrow])

    #modify background count weights, if necessary
    if yback is not None:
        #flag the background counts
        bck = reduce(np.logical_or, [ii == i for i in iback])

        if weightrows is None:
            weight = np.ones([1, cnts.shape[1]])
            weight[bck] = -area_ratio*weight[bck]
            cnts = cnts.append(weight)
        else:
            for i in weightrows:
                cnts[i, bck] = -area_ratio*cnts[i, bck]

    #remove counts outside of all bands
    sig = reduce(np.logical_or, [ii == i for i in isignal])
    keep = sig | bck
    cnts = cnts[:, keep]

    return cnts

def __form_edges(ysignal, yback, counts, yrow):

    # join the edges into one list
    ys = np.array(ysignal) if yback is None else np.append(ysignal, yback)
    edges = np.unique(ys)

    ymax = counts[yrow].max()
    ymin = counts[yrow].min()
    if ymax < edges.max() or ymin > edges.min():
        raise ValueError('Extraction ribbons include areas beyond the range of the counts.')

    # find where signal band is in sorted edges
    isignal = np.searchsorted(edges, mynp.midpts(ysignal))

    if yback is not None:
        # find area ratio of signal to background
        area_signal = ysignal[1] - ysignal[0]
        area_back = np.sum(np.diff(yback, axis=1))
        area_ratio = float(area_signal)/area_back

        # find where background bands are in the sorted edges
        yback_mids = np.reshape(mynp.midpts(yback, axis=1), -1)
        iback = np.searchsorted(edges, yback_mids)
    else:
        iback, area_ratio = None, None

    return edges, isignal, iback, area_ratio

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