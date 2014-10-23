#TODO: merge these two functions
import x1dutils as utils
from numpy import linspace, random
from scipy.interpolate import interp1d

def spectrifyCOS(tag, x1d):
    """Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    
    If there is more than one order, and order array is also added to specify
    which order each photon is likely associated with.
    """
    utils.same_obs([tag, x1d])
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
    """Add spectral units (wavelength, cross dispersion distance, energy/area) 
    to the photon table in the fits data unit "tag".
    
    If there is more than one order, and order array is also added to specify
    which order each photon is likely associated with.
    """
    utils.same_obs([tag, x1d])
    Norders = max(x1d['sci'].data['sporder'])
    Nx_x1d, Ny_x1d = [x1d['primary'].header[key] for key in 
                      ['sizaxis1','sizaxis2']]
                      
    computeEperA = utils.x1d_epera_solution(x1d)
    
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue
        Nx_tag, Ny_tag = [t.header[key] for key in ['axlen1','axlen2']]
        if Norders > 1:
            print 'Wavelength extraction for STIS echelle gratings not yet',
            print 'implemented.'
            return
            
            #TODO: implement
            #here is how you did it in IDL, for when you get around to doing it
            #here
#        ;Loop through all tags, associating each with a dispersion line and finding
#        ;its wavelength.
#        fac = Npixx/FLOAT(Nwav)
#        FOR j = 0, Ntags[i]-1 DO BEGIN
#            ;First, find the y-values of the disperson lines at the x location of the tag
#            ydispl = ydisp[FLOOR(x[j]/fac),*] ;y values to the left of the tag location
#            ydispr = ydisp[CEIL(x[j]/fac),*] ;y values to the right of the tag location
#            ydispi = (ydispr - ydispl)*(x[j]/fac - FLOOR(x[j]/fac))+ydispl ;interpolated y values at the tag location 
#            
#            ;Find the dispersion line with the y value closest to the tag and the 
#            ;distance in y of that tag from the nearest dispersion line
#            dy = y[j] - fac*ydispi ;Hmmmm not sure about that fac
#            yoff[j] = MIN(dy, line, /ABSOLUTE)
#            
#            ;Get data quality flag of the nearest pixel
#            xadj = ROUND(x[j])
#            xadj = xadj > 0
#            xadj = xadj < 1023
#            dq[j] = x1ddq[xadj, line]
#            
#            ;Find the associated wavelength of the tag on its dispersion line
#            wvln[j] = INTERPOL(wavsoln[*,line], pixx, x[j]/fac)
#        ENDFOR
        if Norders == 1:
            t.data['axis1'] += (random.random(t.data['axis1'].shape) - 0.5)
            xpix = linspace(0.0, Nx_tag, Nx_x1d)
            wavinterp = interp1d(xpix, x1d['sci'].data['wavelength'])
            yfac = Ny_tag/Ny_x1d
            extryinterp = interp1d(xpix, x1d['sci'].data['extrlocy']*yfac)
            wave = wavinterp(t.data['axis1'])
            xdisp = (t.data['axis2'] - extryinterp(t.data['axis1']))
            epera = computeEperA(wave[0,:])
            tag[i] = utils.append_cols(t, ['wavelength', 'xdisp', 'epera'], 
                                       ['1E', '1E', '1E'], [wave, xdisp, epera])