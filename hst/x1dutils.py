# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:20:05 2014

@author: parke
"""
from astropy.io import fits
import numpy as np
import os
from scipy.interpolate import interp1d
import pdb

def x1d_wedges(x1d):
    """Reconstructs the wavelength bins used in the x1d."""
    if type(x1d) is str: x1d = fits.open(x1d)
    wave = x1d[1].data['wavelength']
    Norders, Npts = wave.shape
    dwave = np.zeros(wave.shape)
    dwave[:,:-1] = wave[:,1:] - wave[:,:-1]
    dwave[:,-1] = dwave[:,-2]
    wedges = np.zeros([Norders, Npts+1])
    wedges[:,:-1], wedges[:,-1] = wave - dwave/2.0, wave[:,-1] + dwave[:,-1]/2.0
    return wedges

def x1d_epera_solution(x1d):
    """Uses the x1d file to create a function that computes the energy/area
    [erg/cm**2] for a count of a given wavelength and spectral order (row in
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
    wave = x1d[1].data['wavelength']
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
        
def __getfiles(folder, suffix):
    allfiles = os.listdir(folder)
    return filter(lambda s: suffix in s, allfiles)