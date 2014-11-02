#TODO: merge these two functions
import x1dutils as utils
from numpy import random, arange, array, argmin, abs, nan, zeros, ones, floor
from scipy.interpolate import interp1d

def spectrify(tag, x1d):
    """
    Add spectral units to the photon table in the fits data unit "tag".
    
    Added vectors include wavelength, cross dispersion distance from the
    nearest order's dispersion line, energy/area, nearest order number, and
    data quality flags.
    """
    
    utils.same_obs([tag, x1d])
    inst = x1d[0].header['instrume']
    if inst == 'COS':
        spectrifyCOS(tag,x1d)
    elif inst == 'STIS':
        spectrifySTIS(tag,x1d)
    else:
        raise ValueError('Spectrify not implemented for the {} instrument.'.format(inst))
    
def spectrifyCOS(tag, x1d):
    """
    Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    
    If there is more than one order, and order array is also added to specify
    which order each photon is likely associated with.
    """
    #TODO: add order and dq
    segment = tag[0].header['segment'][-1]
    order = 0 if segment == 'A' else 1
    computeEperA = utils.x1d_epera_solution(x1d)
    yexpected, yoff = [x1d[1].header[s+segment] for s in ['SP_LOC_','SP_OFF_']]
    yspec = yexpected + yoff
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue
        xdisp = t.data['yfull'] - yspec
        epera = computeEperA(t.data['wavelength'],order)
        tag[i] = utils.append_cols(t, ['xdisp', 'epera'], ['1E', '1E'], [xdisp, epera])

def spectrifySTIS(tag, x1d):
    """
    Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    
    If there is more than one order, and order array is also added to specify
    which order each photon is likely associated with.
    """
    xd = x1d['sci'].data
    Norders = x1d['sci'].header['naxis2']
    Nx_x1d, Ny_x1d = [x1d[0].header[key] for key in ['sizaxis1','sizaxis2']]
                      
    computeEperA = utils.x1d_epera_solution(x1d)
    
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue
        td = t.data
        
        #change time scale to s
        td['time'] = td['time']*t.header['tscal1']
        del(t.header['tscal1'])
        
        #add random offsets within pixel range to avoid wavelength aliasing
        #issues from quantization
        random.seed(0) #for reproducibility
        x = td['axis1'] + random.random(td['axis1'].shape)
        y = td['axis2'] + random.random(td['axis2'].shape)
        
        #compute interpolation functions for the dispersion line y-value and 
        #the wavelength solution for each order
        Nx_tag, Ny_tag = [t.header[key] for key in ['axlen1','axlen2']]
        xfac, yfac = Nx_tag/Nx_x1d, Ny_tag/Ny_x1d
        xpix = arange(1.0 + xfac/2.0, Nx_tag + 1.0, xfac)
        interp = lambda vec: interp1d(xpix, vec, bounds_error=False, 
                                         fill_value=nan)
        extryinterp = map(interp, xd['extrlocy']*yfac)
        waveinterp = map(interp, xd['wavelength'])
        dqinterp = [interp1d(xpix, dq, 'nearest', bounds_error=False, fill_value=nan)
                    for dq in xd['dq']]
        
        
        if Norders > 1:
            #associate each tag with an order by choosing the closest order
            xdisp = array([y - yint(x) for yint in extryinterp])
            line = argmin(abs(xdisp), 0)
            
            #now get all the good stuff
            xdisp = xdisp[line,arange(len(x))]
            order = xd['sporder'][line]
            #looping through lines is 20x faster than looping through tags
            wave, dq = zeros(x.shape), zeros(x.shape, int)
            for l in range(Norders):
                ind = (line == l)
                wave[ind] = waveinterp[l](x[ind])
                dq[ind] = dqinterp[l](x[ind])
            epera = computeEperA(wave, line)
            
        if Norders == 1:
            dq = xd['dq'][0][tag_x1d_x]
            order = xd['sporder'][0]*ones(x.shape)
            wave = waveinterp[0](x)
            xdisp = (y - extryinterp[0](x))
            epera = computeEperA(wave)
            
        tag[i] = utils.append_cols(t, 
                                   ['wavelength', 'xdisp', 'epera', 'order', 'dq'], 
                                   ['1E', '1E', '1E', '1I', '1I'], 
                                   [wave, xdisp, epera, order, dq])