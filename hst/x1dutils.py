# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:20:05 2014

@author: parke
"""
from astropy.io import fits
import numpy as np
import os
from scipy.interpolate import interp1d
import mypy.my_numpy as mnp
import re
from mypy import specutils

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

def good_waverange(x1d, clipends=False):
    """Returns the range of good wavelengths based on the x1d.
    
    clipends will clip off the areas at the ends of the spectra that have bad
    dq flags.
    """
    xd = fits.getdata(x1d) if type(x1d) is str else x1d[1].data
    wave = xd['wavelength']
    if clipends:
        dq = xd['dq']
        minw, maxw = [], []
        for w,d in zip(wave,dq):
            good = (d == 0)
            minw.append(w[good][0])
            maxw.append(w[good][-1])
        return np.array([minw,maxw]).T
    else:
        return np.array([[wave[0,0], wave[0,-1]], [wave[1,0], wave[1,-1]]])
    
def coadd(x1ds):
    """Coadd spectra from x1d files.
    
    Do this in a "union" sense, keeping all data -- not just the data where the
    spectra overlap. Data that only partially covers a wavelength bin is not
    included in that bin.
    """
    
    if type(x1ds[0]) is str:
        x1dfiles = x1ds
        x1ds = map(fits.open, x1dfiles)
    
    welist = map(wave_edges, x1ds)
    data = [[x1d[1].data[s] for s in ['flux', 'error', 'dq']] for x1d in x1ds]
    f, e, dq = zip(*data)
    t = [x1d[1].header['exptime'] for x1d in x1ds]
    
    return specutils.coadd(welist, f, e, t, dq)

def tagx1dlist(folder,sorted=False):
    allfiles = os.listdir(folder)
    tagfiles = filter(lambda s: 'tag' in s, allfiles)
    obsids = np.unique([f[:9] for f in tagfiles])
    tags, x1ds = [],[]
    for obsid in obsids:
        tag = filter(lambda s: obsid in s, tagfiles)
        if tag == []:
            raise ValueError('No tag files found for observation {}'.format(obsid))
        tags.extend([os.path.join(folder, t) for t in tag])
        x1d = obsid + '_x1d.fits'
        if x1d not in allfiles:
            raise ValueError('No x1d file found for observation {}'.format(obsid))
        x1ds.extend([os.path.join(folder, x1d)]*len(tag))
    return tags,x1ds
    
def tag2x1d(tagname):
    """Determine the corresponding x1d filename from  tag filename."""
    return re.sub('_(corr)?tag_?[ab]?.fits', '_x1d.fits', tagname)

def __getfiles(folder, suffix):
    allfiles = os.listdir(folder)
    return filter(lambda s: suffix in s, allfiles)