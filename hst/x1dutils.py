# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:20:05 2014

@author: parke
"""
from astropy.io import fits
import numpy as np
import os
from scipy.interpolate import interp1d
import my_numpy as mnp
from math import ceil

def wave_edges(x1d):
    """Reconstructs the wavelength bins used in the x1d.
    
    Assumes a linear change in the wavelength step."""
    if type(x1d) is str: x1d = fits.open(x1d)
    ws = x1d[1].data['wavelength']
    es = []
    for w in ws:
        w = w[np.isfinite(w)]
        Npts = len(w)
        dw0 = (w[1] - w[0])
        dw1 = (w[-1] -w[-2])
        dwslope = (dw1 - dw0)/Npts
        dw00 = -0.5*dwslope + dw0
        
        e = np.zeros(Npts+1)
        e[0] = w[0] - dw00/2.0
        for i in np.arange(1,Npts+1): e[i] = 2*w[i-1] - e[i-1]
        es.append(e)
    return es

def x1d_epera_solution(x1d):
    """Uses the x1d file to create a function that computes the energy/area
    [erg/cm**2] for a count of a given wavelength and spectral line (row no. in
    the x1d arrays).
    """
    #get epera from x1d
    if type(x1d) is str: x1d = fits.open(x1d)
    wave, cps, flux = [x1d[1].data[s] for s in ['wavelength', 'net', 'flux']]
    dwave = np.zeros(wave.shape)
    dwave[:,:-1] = wave[:,1:] - wave[:,:-1]
    dwave[:,-1] = dwave[:,-2]
    flux, cps, dwave, wave = map(list, [flux, cps, dwave, wave])
    for i in range(len(flux)):
        keep = (cps[i] != 0)
        flux[i], cps[i], dwave[i], wave[i] = [v[keep] for v in 
                                              [flux[i],cps[i],dwave[i], wave[i]]]
    EperAperCount = [f/c*d for f,c,d in zip(flux,cps,dwave)]
    
    #make an inerpolation function for each order
    intps = [interp1d(w,E,bounds_error=False) for w,E in zip(wave,EperAperCount)]
    
    #the function to be returned. it chooses the proper interpolation function
    #based on the orders of the input counts, then uses it to get epera from
    #wavelength
    def f(waves, orders=0):
        if type(orders) == int: 
            eperas = intps[orders](waves)
        else:
            eperas = np.zeros(waves.shape)
            for i in range(len(intps)):
                pts = (orders == i)
                eperas[pts] = intps[i](waves[pts])
        return eperas
    
    return f

def same_obs(hdus):
    rootnames = [hdu[0].header['rootname'] for hdu in hdus]
    if rootnames.count(rootnames[0]) < len(rootnames):
        raise Exception('The fits data units are from different observations.')

def saveonly(hdu, keys):
    pass

def append_cols(tblhdu, names, formats, arrays):
    """Appends columns to a table hdu and returns the new hdu.
    """
    cols = tblhdu.data.columns
    N = cols[0].array.shape[0]
    for n,f,a in zip(names, formats, arrays):
        if a.shape[0] != N:
            a = np.reshape(a, [N, a.size/N])
        cols.add_col(fits.Column(n, f, array=a))
    rec = fits.FITS_rec.from_columns(cols)
    return fits.BinTableHDU(rec, name=tblhdu.name, header=tblhdu.header)

def wave_overlap(x1dfiles):
    """Returns the overlapping range of wavelengths for each order (or segment) 
    of the spectra in a set of x1d files.
    """
    if type(x1dfiles) == str: x1dfiles = __getfiles(x1dfiles, 'x1d')
    
    #function to read min an max good wavelengths from files 
    minwaves, maxwaves = [], []
    for x1dfile in x1dfiles:
        x1d = fits.open(x1dfile)
        rngs = good_waverange(x1d)
        minwaves.append(rngs[:,0])
        maxwaves.append(rngs[:,1])
        x1d.close()
        del x1d
    
    orders = map(len, minwaves)
    if orders.count(orders[0]) < len(orders):
        raise ValueError('All spectral data must contain the same number of '
                         'orders (or segments)). I.e. the wavelengths arrays '
                         'must have the same number of rows.')
    else:
#        minwaves, maxwaves = np.array(minwaves), np.array(maxwaves)
        minwaves, maxwaves = np.max(minwaves, 0), np.min(maxwaves, 0)
        return np.array([minwaves,maxwaves]).T

def good_waverange(x1d):
    """Returns the range of good wavelengths based on the data quality flags
    for each order in the x1d.
    """
    if type(x1d) is str: x1d = fits.open(x1d)
    wave = x1d[1].data['wavelength'].copy()
    dq = x1d[1].data['dq']
    minw, maxw = [], []
    for w,d in zip(wave,dq):
        bad = (d != 0)
        w[bad] = np.nan
        minw.append(np.nanmin(w))
        maxw.append(np.nanmax(w))
    if type(x1d) is str:
        x1d.close()
        del x1d
    return np.array([minw,maxw]).T

def coadd(x1ds):
    """Coadd spectra from x1d files.
    
    Do this in a "union" sense, keeping all data -- not just the data where the
    spectra overlap. Data that only partially covers a wavelength bin is not
    included in that bin.
    """
    
    if type(x1ds[0]) is str:
        x1dfiles = x1ds
        x1ds = map(fits.open, x1dfiles)
    
    #specify a master grid on which to interpolate the data
    #get the edges, centers, and spacings for each x1d wavegrid 
    welist = map(wave_edges, x1ds)
    welist = reduce(lambda x,y: list(x)+list(y), welist)
    dwlist = [we[1:] - we[:-1] for we in welist]
    wclist = map(mnp.midpts, welist)
    wmin, wmax = np.min(welist), np.max(welist)
    
    #splice the spacings and centers into big vectors
    dw_all, wc_all = dwlist[0], wclist[0]
    for dw, wc in zip(dwlist[1:], wclist[1:]):
        i = np.digitize(wc, wc_all)
        dw_all = np.insert(dw_all, i, dw)
        wc_all = np.insert(wc_all, i, wc)
        
    #identify the relative maxima and make an interpolation function between them
    _,imax = mnp.argextrema(dw_all)
    wcint = np.hstack([[wmin],wc_all[imax],[wmax]])
    dwint = dw_all[np.hstack([[0],imax,[-1]])]
    dwf = interp1d(wcint, dwint)
    
    #construct a vector by beginning at wmin and adding the dw amount specified
    #by the interpolation of the maxima
    w = np.zeros(ceil((wmax-wmin)/np.min(dw)))
    w[0] = wmin
    n = 1
    while True:
        w[n] = w[n-1] + dwf(w[n-1])
        if w[n] > wmax: break
        n += 1
    w = w[:n]
    
    #coadd the spectra (the m prefix stands for master)
    mins, mvar, mexptime = [np.zeros(n-1)for i in range(3)]
    #loop through each order of each x1d
    for x1d in x1ds:
        flux_arr, err_arr, dq_arr = [x1d[1].data[s] for s in 
                                     ['flux', 'error', 'dq']]
        we_arr = wave_edges(x1d)
        exptime = x1d[1].header['exptime']
        for flux, err, dq, we in zip(flux_arr, err_arr, dq_arr, we_arr):
            #intergolate and add flux onto the master grid
            dw = we[1:] - we[:-1]
            flux, err = flux*dw, err*dw
            wrange = [np.min(we), np.max(we)]
            badpix = (dq != 0)
            flux[badpix], err[badpix] = np.nan, np.nan
            overlap = (np.digitize(w, wrange) == 1)
            addins = exptime*mnp.rebin(w[overlap], we, flux)
            addvar = exptime**2*mnp.rebin(w[overlap], we, err**2)
            
            addtime = np.ones(addins.shape)*exptime
            badbins = np.isnan(addins)
            addins[badbins], addvar[badbins], addtime[badbins] = 0.0, 0.0, 0.0
            i = np.nonzero(overlap)[0][:-1]
            mins[i] += addins
            mvar[i] += addvar
            mexptime[i] += addtime
    
    mdw = w[1:] - w[:-1]
    mflux = mins/mexptime/mdw
    merr = np.sqrt(mvar)/mexptime/mdw
    badbins = (mexptime == 0.0)
    mflux[badbins], merr[badbins] = np.nan, np.nan
    return w, mflux, merr, mexptime

def tagx1dlist(folder,sorted=False):
    allfiles = os.listdir(folder)
    obsids = [f[:9] for f in allfiles]
    tags, x1ds = [],[]
    for obsid in obsids:
        tag = filter(lambda s: obsid in s, allfiles)
        if tag == []:
            raise ValueError('No tag files found for observation {}'.format(obsid))
        tags.extend(tag)
        x1d = obsid + '_x1d.fits'
        if x1d not in allfiles:
            raise ValueError('No x1d file found for observation {}'.format(obsid))
        x1ds.extend([x1d]*len(tag))
    return tags,x1ds

def __getfiles(folder, suffix):
    allfiles = os.listdir(folder)
    return filter(lambda s: suffix in s, allfiles)