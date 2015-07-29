import x1dutils as utils
from numpy import random, arange, array, argmin, abs, nan, zeros, ones, median, var, polyfit, polyval, inf
from scipy.interpolate import interp1d
from mypy.my_numpy import divvy, midpts

def spectrify(tag, x1d, traceloc='stsci'):
    """
    Add spectral units to the photon table in the fits data unit "tag".
    
    Added vectors include wavelength, cross dispersion distance from the
    nearest order's dispersion line, energy/area, nearest order number, and
    data quality flags.
    """
    
    utils.same_obs([tag, x1d])
    inst = x1d[0].header['instrume']
    if inst == 'COS':
        spectrifyCOS(tag,x1d, traceloc)
    elif inst == 'STIS':
        spectrifySTIS(tag,x1d, traceloc)
    else:
        raise ValueError('Spectrify not implemented for the {} instrument.'.format(inst))
    
def spectrifyCOS(tag, x1d, traceloc='stsci'):
    """
    Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    """
    #TODO: check to see if this works with g230l
    segment = tag[0].header['segment'][-1]
    order = 0 if segment == 'A' else 1
    computeEperA = utils.x1d_epera_solution(x1d)
    
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue

        td,th = t.data, t.header

        if traceloc == 'stsci':
            """
            Note: How STScI extracts the spectrum is unclear. Using 'y_lower/upper_outer' from the x1d reproduces the
            x1d gross array, but these results in an extraction ribbon that has a varying height and center -- not
            the parallelogram that is described in the Data Handbook as of 2015-07-28. The parameters in the
            xtractab reference file differ from those populated in the x1d header. So, I've punted and stuck with
            using the x1d header parameters because it is easy and I think it will make little difference for most
            sources. The largest slope listed in the xtractab results in a 10% shift in the spectral trace over the
            length of the detector. In general, I should just check to be sure the extraction regions I'm using are
            reasonable.
            """
            yexpected, yoff = [x1d[1].header[s+segment] for s in 
                               ['SP_LOC_','SP_OFF_']]
            yspec = yexpected + yoff
            xdisp = td['yfull'] - yspec
        if traceloc == 'median':
            Npixx  = th['talen2']
            x, y = td['xfull'], td['yfull']
            xdisp = __median_trace(x, y, Npixx, 8)
        if traceloc == 'lya':
            Npixy = th['talen3']
            xdisp = __lya_trace(td['wavelength'], td['yfull'], Npixy)
        epera = computeEperA(td['wavelength'],order)
        tag[i] = utils.append_cols(t, ['xdisp', 'epera'], ['1D']*2, [xdisp, epera])

def spectrifySTIS(tag, x1d, traceloc='stsci'):
    """
    Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    
    If there is more than one order, an order array is also added to specify
    which order each photon is likely associated with.
    """
    xd = x1d['sci'].data
    Norders = x1d['sci'].header['naxis2']
    Nx_x1d, Ny_x1d = [x1d[0].header[key] for key in ['sizaxis1','sizaxis2']]
    
    if Norders > 1 and traceloc != 'stsci':
        raise NotImplemented('Cannot manually determine the spectral trace '
                             'locations on an echellogram.')
    
    computeEperA = utils.x1d_epera_solution(x1d)
    
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue
        td, th = t.data, t.header

        #change time scale to s
        td['time'] = td['time']*th['tscal1']
        del(th['tscal1'])

        x,y = td['axis1'],td['axis2']
        #there seem to be issues with at the stsci end with odd and even
        #pixels having systematically different values (at least for g230l)
        #so group them by 2-pixel
        xeven, yeven = (x % 2 == 1), (y % 2 == 1)
        x[xeven] = x[xeven] - 1
        y[yeven] = y[yeven] - 1
        
        #add random offsets within pixel range to avoid wavelength aliasing
        #issues from quantization
        random.seed(0) #for reproducibility
        x = x + random.random(x.shape)*2.0
        y = y + random.random(y.shape)*2.0
        
        #compute interpolation functions for the dispersion line y-value and 
        #the wavelength solution for each order
        Nx_tag, Ny_tag = th['axlen1'], th['axlen2']
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
            dq = dqinterp[0](x)
            order = xd['sporder'][0]*ones(x.shape)
            wave = waveinterp[0](x)
            if traceloc == 'stsci':
                xdisp = (y - extryinterp[0](x))
            if traceloc == 'median':
                xdisp = __median_trace(x, y, Nx_tag)
            if traceloc == 'lya':
                xdisp = __lya_trace(wave, y, Ny_tag)
            epera = computeEperA(wave)
        
        newcols = ['wavelength', 'xdisp', 'epera', 'order', 'dq']
        dtypes = ['1D']*3 + ['1I']*2
        data = [wave, xdisp, epera, order, dq]
        tag[i] = utils.append_cols(t, newcols, dtypes, data)
    
def __median_trace(x, y, Npix, binfac=1):
    #NOTE: I looked into trying to exclude counts during times when the 
    #calibration lamp was on for COS, but this was not easily possible as of 
    #2014-11-20 because the lamp flashes intermittently and the times aren't
    #recorded in the corrtag files
    
    # get the median y value and rough error in each x pixel
    cnts = array([x,y])
    bins = arange(0,Npix+1, binfac)
    binned = divvy(cnts, bins)
    binned = [b[1] for b in binned]
    meds = array(map(median, binned))
    sig2 = array(map(var, binned))
    Ns = array(map(len, binned))
    sig2[Ns <= 1] = inf
    ws = Ns/sig2
    
    # fit a line and subtrqact it from the y values
    p = polyfit(midpts(bins), meds, 1, w=ws)
    return y - polyval(p, x)
    
def __lya_trace(w, y, ymax):
    cnts = array([w,y])
    lya_range = [1214.5,1217.2]
    lyacnts, = divvy(cnts, lya_range)
    ytrace_old = inf
    ytrace = median(lyacnts[1])
    dy = ymax/2.0
    #iterative narrow down the yrange to block out the distorted airglow
    while abs(ytrace - ytrace_old)/ytrace > 1e-4:
        dy *= 0.5
        lyacnts, = divvy(lyacnts, [ytrace-dy, ytrace+dy], 1)
        ytrace_old = ytrace
        ytrace = median(lyacnts[1])
    return y - ytrace