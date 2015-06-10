# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:31:42 2014

@author: Parke
"""
from astropy.io import fits
import x1dutils
from spectrify import spectrify
import numpy as np
import mypy.my_numpy as mnp
from astropy.table import Table, Column
from os import path
from .. import functions as sp
from warnings import warn
from mypy.specutils import common_grid
from time import strftime

def autocurve(tagfiles, x1dfiles, dt, waste=True, bands='broad', groups=None,
              extrsize='stsci', bksize='stsci', bkoff='stsci', contbands=None,
              contorder=None, silent=False, fitsout=None, clobber=True):
    """
    Generates a lightcurve from the provided _tag and _x1d FITS files.

    Not all files need cover the appropriate wavlengths (the function will
    select appropriately).

    Parameters
    ----------
    dt : float
        Time step to use.
    waste : {True|False}, optional
        If false, take dt to be a suggestion rather than a rule and adjust the
        time step for each exposure such that all of the exposure can be used.
        If true (the default), treat the time step as strict and drop any
        fractional remainder at the end of an exposure.
    bands : {list|array|'broad'}, optional
        Specifies the wavelength ranges to integrate over. An Nx2 (or 2xN) list or
        array specifies the start and end value for N bands. For a 1-D list or array
        each successive pair of values are considered a band. Specifying 'braod'
        causes the function to use the largest range of overlap between the x1d
        files as bands.
    extrsize : {float|'stsci'}, optional
        The half-width of the region centered on the spectral drace from which to extract
        signal counts (the width of the signal "ribbon").
    bksize : {array|list|float|'stsci'}, optional
        As with extrsize, except for the background regions. An array or list
        may be used to specify multiple extraction regions.
    bkoff : {array|list|float|'stsci'}, optional
        The distance above and below the signal extraction ribbion to center
        the background extraction ribbons. An array or list
        may be used to specify multiple extraction regions. If only a single
        value is given, it will be used to creat two regions, one above and
        one below the spectral trace.
    fitsout : {str|list}, optional
        Filenames for writing lightcurves. There must be one name per group
        (or one per band if groups == None).

    """
    #FIXME: make this work for g230l as well

    # -----GROOM INPUT-----
    if bands == 'broad':
        bands = x1dutils.wave_overlap(x1dfiles)
    bands = np.asarray(bands)
    if bands.ndim == 1: bands = np.reshape(bands, [len(bands)/2, 2])
    if bands.shape[1] != 2 and bands.shape[0] == 2: bands = bands.swapaxes(0,1)

    #divide the files into a and b lists
    tagafiles = filter(lambda f: 'corrtag_b' not in f, tagfiles)
    tagbfiles = filter(lambda f: 'corrtag_b' in f, tagfiles)

    #sort the tagfiles by exposure start time
    tagafiles.sort(key=lambda tf: fits.getval(tf, 'expstart', ext=1))

    #get reference mjd
    mjdref = fits.getval(tagfiles[0], 'expstart', ext=1)

    if groups is None: groups = [range(bands.shape[0])]
    Ncurves = len(groups)

    contsub = (contorder is not None)

    if not hasattr(fitsout, '__iter__'): fitsout = [fitsout]

    # -----INITIALIZE-----
    inst, optel, exp, segvec = map(list, np.empty([4,Ncurves,0],str))
    t0, t1, net, neterr, flux, fluxerr = map(list, np.empty([6,Ncurves,0]))

    if contsub:
        cont, conterr, line, linerr, fitcens = map(list, np.empty([5,Ncurves,0]))
        Nfit = contorder+1
        fitcoeffs = list(np.empty([Ncurves,0,Nfit]))
        fitcovars = list(np.empty([Ncurves,0,Nfit**2]))

    # -----LOOP THROUGH FILES-----
    for i, tagafile in enumerate(tagafiles):
        if not silent: print 'Processing tag file(s) {} of {}'.format(i+1,len(tagafiles))
        x1dfile = __findx1d(tagafile, x1dfiles)
        tagbfile = __findtagb(tagafile, tagbfiles)

        x1d = fits.open(x1dfile)
        taga = fits.open(tagafile)
        tagb = fits.open(tagbfile) if tagbfile else None

        cos = __iscos(x1d)
        if tagbfile:
            seg = 'A+B'
        else:
            seg = 'A' if cos else 'n/a'

        insti, opteli, expi = [x1d[0].header[s] for s in
                               ['instrume','opt_elem','rootname']]

        #determine which lines we can extract from the exposure
        wr = x1dutils.good_waverange(x1d)
        wr = np.sort(wr.ravel())
        leftout = ~mnp.inranges(bands[:,0], wr)
        rightout = ~mnp.inranges(bands[:,1], wr)
        badlines = np.logical_or(leftout, rightout)
        isbadgrp = lambda g: any(badlines[g])
        badgroups = map(isbadgrp, groups)
        if all(badgroups):
            print ('\tNo lines/groups covered by exposure {}, {}'
                   ''.format(i+1,expi))
            continue
        else: #might as well narrow down the wavelength range
            goodbands = bands[np.logical_not(badlines)]
            wr = [np.min(goodbands), np.max(goodbands)]

        #get what we need out of the fits files
        ysignal, yback = __ribbons(x1d, extrsize, bksize, bkoff, 'A')
        cnts = __parsetags(taga, x1d, wr, ysignal, yback)
        toff, tr, tbins, mjdstart = __tinfo(taga, mjdref, dt, waste)
        if tagb:
            ysignal, yback = __ribbons(x1d, extrsize, bksize, bkoff, 'B')
            bcnts = __parsetags(tagb, x1d, wr, ysignal, yback)
            cnts = np.hstack([cnts, bcnts])
        t,w,f = cnts[:3]
        e = cnts[3] if cos else None

        x1d.close()
        taga.close()
        if tagb: tagb.close()

        #compute the line lightcurves
        def makecurve(bands,e,dt,g):
            return sp.spectral_curves(t, w, bands=bands, tbins=dt, eps=e,
                                      trange=tr, groups=g)
        tedges, cps, cpserr = makecurve(bands, e, tbins, groups)
        _, fx, fxerr = makecurve(bands, f, tbins, groups)
        if contsub:
            _, cfx, cfxerr = makecurve(contbands, f, tbins, None)
            cfx, cfxerr = np.array(cfx).T, np.array(cfxerr).T

        for i,g in enumerate(groups):
            if badgroups[i]: continue

            #lightcurves
            Npts = len(cps[i])
            inst[i], optel[i], exp[i] = [np.append(x,[y]*Npts) for x,y in
                                         [[inst[i], insti], [optel[i],opteli],
                                          [exp[i],expi]]]
            segvec[i] = np.append(segvec[i], [seg]*Npts)
            t0[i] = np.append(t0[i], tedges[i][:-1]+toff)
            t1[i] = np.append(t1[i], tedges[i][1:]+toff)
            net[i] = np.append(net[i], cps[i])
            neterr[i] = np.append(neterr[i], cpserr[i])
            flux[i] = np.append(flux[i], fx[i])
            fluxerr[i] = np.append(fluxerr[i], fxerr[i])

            #continuum subtraction
            if contsub:
                wmid = np.mean(bands[g])
                tempcf, tempcfe = np.zeros(Npts), np.zeros(Npts)
                coeffs = np.zeros([Npts,Nfit])
                covars = np.zeros([Npts,Nfit**2])
                for j, [c, ce] in enumerate(zip(cfx, cfxerr)):
                    contfit = mnp.polyfit_binned(contbands - wmid, c, ce,
                                                 contorder)
                    covars[j] = contfit[1].ravel()
                    coeffs[j], fitfun = contfit[0], contfit[2]
                    fitflux, fiterr = fitfun(bands[g] - wmid)
                    tempcf[j] = np.sum(fitflux)
                    tempcfe[j] = np.sqrt(np.sum(fiterr**2))
                fitcoeffs[i] = np.vstack([fitcoeffs[i], coeffs])
                fitcovars[i] = np.vstack([fitcovars[i], covars])
                fitcens[i] = np.append(fitcens[i], wmid*np.ones(Npts))
                cont[i] = np.append(cont[i], tempcf)
                conterr[i] = np.append(conterr[i], tempcfe)
                line[i] = np.append(line[i], fx[i] - tempcf)
                linerr[i] = np.append(linerr[i], np.sqrt(fxerr[i]**2 +
                                      tempcfe**2))

    # -----MAKE TABLES-----
    names = ['instrume', 'opt_elem','segment','exposure','t0','t1','cps',
             'cps err','flux','flux err']
    units = ['']*4 + ['s','s','counts/s','counts/s'] + ['erg/(s*cm**2)']*2
    descriptions = ['instrument',
                    'optical element (grating) used',
                    'detector segement (applies only to COS)',
                    'observation/exposure identifier (9 letter string used in the stsci filenames)',
                    'bin start time',
                    'bin end time',
                    'background subtracted count rate (counts per second)',
                    'count rate error',
                    'background subtracted flux',
                    'flux error']
    if contsub:
        dataset = zip(inst, optel, segvec, exp, t0, t1, net, neterr, flux,
                      fluxerr, cont, conterr, line, linerr, fitcens, fitcoeffs,
                      fitcovars)
        names.extend(['cont flux', 'cont flux err', 'line flux','line flux err',
                      'cont fit center','cont fit coeffs','cont fit covars'])
        units.extend(['erg/(s*cm**2)']*4 + ['angstrom'] + ['',''])
        descriptions.extend(['continuum flux estimate over the line bandpass',
                             'error on the continuum flux estimate',
                             'line flux estimate (flux - cot flux)',
                             'error on the line flux estimate',
                             'central wavelength used for the polynomial continuum fit',
                             'polynomial fit cofficients (flux units), highest power first',
                             'elements of the polynomial fit covariance matrix, use np.reshape(N,N) to retrieve the matrix'])
    else:
        dataset = zip(inst, optel, segvec, exp, t0, t1, net, neterr, flux,
                  fluxerr)

    tables = []
    for data in dataset:
        cols = [Column(d,n,unit=u,description=dn) for d,n,u,dn in
                zip(data,names,units,descriptions)]
        tbl = Table(cols)
        tbl.meta['mjd0'] = mjdref
        tables.append(tbl)

    # -----WRITE TO FITS-----
    if fitsout is not None:
        fmts = ['4A','5A','3A','9A'] + ['E']*6
        if contsub:
            fmts.extend(['E']*5 + ['{}E'.format(Nfit), '{}E'.format(Nfit**2)])

        for i,[fitsfile,data] in enumerate(zip(fitsout,dataset)):
            if data[0].size == 0:
                warn('The exposures did not cover line/group {}. No '
                     'lightcurve will be written to a FITS file for '
                     'that line.'.format(i))
                continue

            #lightcurve hdu
            cols = [fits.Column(n,fm,u,array=d) for n,u,fm,d in
                    zip(names,units,fmts,data)]
            rec = fits.FITS_rec.from_columns(cols)
            curvehdu = fits.BinTableHDU(rec, name='lightcurve')
            if contsub:
                curvehdu.header['comment'] = ('The continuum fit is privided by '
                'giving the central wavelength used for the fit (cont fit center), '
                'the N coffecients of the Nth order polynomial (aN,...,a1,a0; '
                'cont fit coeffs), and the NxN covariance matrix for the fit '
                'errors given as a 1xN**2 vector such that the first row is '
                'entries 0:N, the second is N:2N, and so forth (cont fit covars).')

            #primary hdu
            prihdr = fits.Header()
            prihdr['mjdstart'] = mjdref
            prihdr['comment'] = ('This file contains lightcurve data extracted from one or more HST observations. '
                                 'The second extension gives the flux vs time data. '
                                 'The third extension gives the wavelength ranges '
                                 'over which signal flux was integrated. It also '
                                 'gives the ranges used to fit continuum flux if applicable. '
                                 'The HST exposures '
                                 'with the original data are idenifyable via '
                                 'the instrume, opt_elem, and exposure columns of '
                                 'the table in the second extension.')
            prihdu = fits.PrimaryHDU(header=prihdr)

            #band wavelengths
            w0,w1 = bands[groups[i]].T
            names2 = ['wave0','wave1']
            fmts2 = ['E']*2
            units2 = ['angstrom']*2
            data2 = [w0,w1]
            if contsub:
                cw0,cw1 = contbands.T
                names2.extend(['contwave0','contwave1'])
                data2.extend([cw0,cw1])
                fmts2.extend(['E']*2)
                units2.extend(['angstrom']*2)

            cols = [fits.Column(n,fm,u,array=d) for n,fm,u,d in
                    zip(names2, fmts2, units2, data2)]
            rec = fits.FITS_rec.from_columns(cols)
            wavehdu = fits.BinTableHDU(rec, name='bandpasses')

            hdu = fits.HDUList([prihdu,curvehdu,wavehdu])

            # write it
            hdu.writeto(fitsfile, clobber=clobber)

    return tables

__iscos = lambda x1d: x1d[0].header['instrume'] == 'COS'
__isstis = lambda x1d: x1d[0].header['instrume'] == 'STIS'

def autospec(tagfiles, x1dfiles, wbins='stsci', traceloc='stsci', extrsize='stsci',
             bksize='stsci', bkoff='stsci', fitsout=None, clobber=True):
    """
    Creates a spectrum starting with the counts.

    Background counts are treated by weighting them by the ratio of the signal
    to background area, allowing different signal and background reagions to
    be used between files.

    Parameters
    ----------
    traceloc : {'stsci'|'median'|'lya'|float}
        How to identify the location of the spectra trace.
        'stsci' : use the location given by stsci in the x1d
        'median' : fit a line to the median y value of the counts as a function
            of x
        'lya' : use a constant value set at the median y-value of counts just
            falling in the Lyman-a line
        float : user-specified constant value

    Returns
    -------
    spec : astropy table
        A table representing the spectrum. Look within the table for
        column descriptions, units, etc.

    Cautions
    --------
    Flat fielding is not properly implemented for creating spectra from tags.
        The STScI x1d is used to compute fluxes from count rates, but the x1d
        only compesates for flat-field features within the signal and
        extraction regions used to generate the x1d. If significantly different
        regions are used by this function to generate a spectrum, systematic
        flat-field errors will result.
    Combining data from different instruments will make count rates
        meaningless and fluxes questionable.
    Bad data quality areas are not masked out.
    """
    #TODO: test with multiple files


    # -----PARSE DATA-----
    #parse out the counts and wavelength grids from each pair of files
    signallist, backlist = [], [] #for counts
    exptimes = []
    mjdranges, configs, obsids = [], [], [] #meta info for the spectrum source
    welist = [] #for wavelength grids
    for tagfile in tagfiles:
        x1dfile = __findx1d(tagfile, x1dfiles)

        #open the files
        tag, x1d = map(fits.open, [tagfile, x1dfile])

        #extract important header info
        keys = ['instrume','opt_elem','cenwave','rootname']
        inst, optel, cenwave, obsid = [tag[0].header[key] for key in keys]
        mjdranges.append([tag[1].header[key] for key in ['expstart','expend']])
        exptimes.append(tag[1].header['exptime'])

        #if there is segment information, get it
        try: seg = tag[0].header['segment']
        except KeyError: seg = ''

        #record onfiguration string and obsid
        obsids.append(obsid)
        configs.append(' '.join([inst, optel, seg, str(cenwave)]))

        #get the wavelength grid
        wmids = x1d[1].data['wavelength']
        #keep only the row corresponding to the current tag file, if necessary
        if seg != '':
            xsegs = list(x1d[1].data['segment'])
            i = xsegs.index(seg)
            wmids = wmids[[i]]
        #convert to bin edges
        we = [mnp.mids2edges(wm, 'left', 'linear-x') for wm in wmids]
        welist.extend(we) #store

        #add spectral information to the counts
        spectrify(tag, x1d, traceloc)

        #set the signal and background extraction regions
        ysignal, yback = __ribbons(x1d, extrsize, bksize, bkoff, seg)

        #divvy up the counts
        cnts = [tag[1].data[s] for s in ['xdisp','wavelength','epera']]
        try: cnts.append(tag[1].data['epsilon']) #if there is weight information...
        except KeyError: cnts.append(np.ones(len(cnts[0])))
        cnts = np.array(cnts)
        signal, back, ar = sp.divvy_counts(cnts, ysignal, yback)

        #weight the background counts by the area ratio
        back[-2:] *= ar

        #store in lists
        signallist.append(signal[1:])
        backlist.append(back[1:])

    #-----MAKE SPECTRA-----
    #merge the arrays and parse out counts
    signal, back = map(np.hstack, [signallist, backlist])
    w, f, e = signal
    wb, fb, eb = back

    #make grid (need it to get exposure times)
    if wbins == 'stsci':
        wbins = common_grid(welist)
        wrange = wbins[[0,-1]]
    elif hasattr(wbins, '__iter__'):
        wrange = wbins[[0,-1]]
    else:
        wrange = [np.min(w), np.max(w)]
        if type(wbins) is float:
            wbins = np.arange(wrange[0], wrange[1], wbins)
        elif type(wbins) is int:
            wbins = np.linspace(wrange[0], wrange[1], wbins)
    dw = np.diff(wbins)

    #compute exposure time within each bin
    N = len(wbins) - 1
    exptbinned = np.zeros(N)
    for exptime, we in zip(exptimes, welist):
        # pull out all we that at least partially overlap wbins
        add_in = mnp.binoverlap(wbins, we, 'loose')
        arg_in, = np.nonzero(add_in)
        arg_in = np.append(arg_in, arg_in[-1] + 1)
        add_we = we[arg_in].astype('f8')
        Nin = len(add_we)

        # create exposure time vector for those bins
        add_expt = np.array([exptime] * (Nin-1), 'f8')

        # add dummy 0.0 exptime bins on both ends
        add_we = np.insert(add_we, [0, Nin], [0.0, wbins[-1] + 1.0])
        add_expt = np.insert(add_expt, [0, Nin-1], [0.0, 0.0])

        # rebin to wbins. wbins outside of we will just get 0.0 because of
        # the dummy zeros we padded with
        add_expt = mnp.rebin(wbins, add_we, add_expt, 'avg')

        # add the result
        exptbinned += add_expt

    #make spectra
    wbins, cpw, cpw_err = sp.spectrum(w, wb, e, eb, 1.0, wbins, wrange=wrange) #counts/AA
    _, flu, flu_err = sp.spectrum(w, wb, f, fb, 1.0, wbins, wrange=wrange) #fluence (ergs/cm2/AA)

    #normalize by time, bin size
    cps, cps_err = cpw*dw/exptbinned, cpw_err*dw/exptbinned
    flux, flux_err = flu/exptbinned, flu_err/exptbinned

    #-----PUT INTO TABLE-----
    #make data columns
    colnames = ['w0','w1','net','net_error','flux','error','exptime']
    units = ['Angstrom']*2 + ['counts/s']*2 + ['ergs/s/cm2/Angstrom']*2 + ['s']
    descriptions = ['left (short,blue) edge of the wavelength bin',
                    'right (long,red) edge of the wavelength bin',
                    'count rate within the bin',
                    'error on the count rate',
                    'average flux over the bin',
                    'error on the flux',
                    'cumulative exposure time for the bin']
    dataset = [wbins[:-1], wbins[1:], cps, cps_err, flux, flux_err, exptbinned]
    cols = [Column(d,n,unit=u,description=dn) for d,n,u,dn in
            zip(dataset, colnames, units, descriptions)]

    #make metadata dictionary
    descriptions = {'mjdranges' : 'Mean-Julian Date start and end of each '
                                  'included exposure',
                    'configurations' : 'the instrument configuraiton of each '
                                       'included exposure, given as '
                                       '\'instrument optical_element segment '
                                       'cenwave\'',
                    'exposures' : 'STScI identifiers of the included exposures'}
    meta = {'descriptions' : descriptions,
            'mjdranges' : mjdranges,
            'configurations' : configs,
            'exposures' : obsids}

    #put into table
    tbl  = Table(cols, meta=meta)

    #-----PUT INTO FITS-----
    if fitsout is not None:
        #spectrum hdu
        cols = [fits.Column(n,'E',u,array=d) for n,u,d in
                zip(colnames, units, dataset)]
        spechdu = fits.BinTableHDU.from_columns(cols, name='spectrum')

        #metadata hdu
        mjdranges = np.array(mjdranges)
        colnames = ['mjdstart','mjdend','configs','exposures']
        units = ['mean julian day']*2 + ['']*2
        maxlen = max(map(len, configs))
        fmts = ['E']*2 + ['{}A'.format(maxlen), '9A']
        dataset = [mjdranges[:,0], mjdranges[:,1], configs, obsids]
        cols = [fits.Column(n,fmt,u,array=d) for n,fmt,u,d in
                zip(colnames, fmts, units, dataset)]
        metahdu = fits.BinTableHDU.from_columns(cols, name='meta')

        #make primary header
        prihdr = fits.Header()
        prihdr['comment'] = ('Spectrum generated from the observations listed '
                             'in the meta hdu, integrated from the count level. '
                             'Created with spectralPhoton software '
                             'http://github.com/parkus/spectralPhoton')
        prihdr['date'] = strftime('%c')
        prihdu = fits.PrimaryHDU(header=prihdr)

        hdulist = fits.HDUList([prihdu,spechdu,metahdu])
        hdulist.writeto(fitsout, clobber=clobber)

    return tbl

def x2dspec(x2dfile, traceloc='max', extrsize='stsci', bksize='stsci',
            bkoff='stsci', x1dfile=None, fitsout=None, clobber=True,
            bkmask=0):
    """
    Creates a spectrum using the x2d file.

    Parameters
    ----------
    x2dfile : str
        Path of the x2d file.
    traceloc : {int|'max'}, optional
        Location of the spectral trace.
        int : the midpoint pixel
        'max' : use the mean y-location of the pixel with highest S/N
    extrsize, bksize, bkoff : {int|'stsci'}, optional
        The height of the signal extraction region, the height of the
        background extraction regions, and the offset above and below the
        spectral trace at which to center the background extraction regions.
        'stsci' : use the value used by STScI in making the x1d (requires
            x1dfile)
        int : user specified value in pixels
    x1dfile : str, optional if 'stsci' is not specfied for any other keyword
        Path of the x1d file.
    fitsout : str, optional
        Path for saving a FITS file version of the spectrum.
    clobber : {True|False}, optional
        Whether to overwrite the existing FITS file.
    bkmask : int, optional
        Data quality flags to mask the background. Background pixels that have
        at least one of these flags will be discarded.

    Returns
    -------
    spectbl : astropy table
        The wavelength, flux, error, and data quality flag values of the extracted
        spectrum.

    Cautions
    --------
    Using a non-stsci extraction size will cause a systematic error because a
    flux correction factor is applied that assumes the STScI extraction
    ribbon was used.

    This still isn't as good as an x1d, mainly because the wavelength dependency
    of the slit losses is not accounted for.
    """
    x2d = fits.open(x2dfile)

    #get the flux and error from the x2d
    f,e,q = x2d['sci'].data, x2d['err'].data, x2d['dq'].data

    inst = x2d[0].header['instrume']
    if inst != 'STIS':
        raise NotImplementedError("This function cannot handle {} data at "
                                  "present.".format(inst))

    #make sure x1d is available if 'stsci' is specified for anything
    if 'stsci' in [traceloc, extrsize, bksize, bkoff]:
        try:
            x1d = fits.open(x1dfile)
            xd = x1d[1].data
        except:
            raise ValueError("An open x1d file is needed if 'stsci' is "
                             "specified for any of the keywords.")

    #select the trace location
    if traceloc == 'max':
        sn = f/e
        sn[q > 0] = 0.0
        sn[e <= 0.0] = 0.0
        maxpixel = np.nanargmax(sn)
        traceloc = np.unravel_index(maxpixel, f.shape)[0]

    #get the ribbon values
    if extrsize == 'stsci': extrsize = np.mean(xd['extrsize'])
    if bksize == 'stsci': bksize = np.mean([xd['bk1size'], xd['bk2size']])
    if bkoff == 'stsci':
        bkoff = np.mean(np.abs([xd['bk1offst'], xd['bk2offst']]))

    #convert everything to integers so we can make slices
    try:
        intrnd = lambda x: int(round(x))
        traceloc, extrsize, bksize, bkoff = map(intrnd, [traceloc, extrsize,
                                                         bksize, bkoff])
    except ValueError:
        raise ValueError("Invalid input for either traceloc, extrsize, bksize, "
                         "or bkoff. See docstring.")

    #convert intensity to flux
    fluxfac = x2d['sci'].header['diff2pt']
    f, e = f*fluxfac, e*fluxfac

    #get slices for the ribbons
    sigslice = slice(traceloc - extrsize//2, traceloc + extrsize//2 + 1)
    bk0slice = slice(traceloc - bkoff - bksize//2, traceloc - bkoff + bksize//2 + 1)
    bk1slice = slice(traceloc + bkoff - bksize//2, traceloc + bkoff + bksize//2 + 1)
    slices = [sigslice, bk0slice, bk1slice]

    #mask bad values in background regions
    if bkmask:
        badpix = (q & bkmask) > 0
        badpix[sigslice] = False #but don't modify the signal region
        f[badpix], e[badpix], q[badpix] = 0.0, 0.0, 0
        #make a background area vector to account for masked pixels
        goodpix = ~badpix
        bkareas = [np.sum(goodpix[slc,:], 0) for slc in slices[1:]]
        bkarea = sum(bkareas)
    else:
        bkarea = bksize*2

    #sum fluxes in each ribbon
    fsig, fbk0, fbk1 = [np.sum(f[slc,:], 0) for slc in slices]

    #sum errors in each ribbon
    esig, ebk0, ebk1 = [mnp.quadsum(e[slc,:], 0) for slc in slices]

    #condense dq flags in each ribbon
    bitor = lambda a: reduce(lambda x,y: x | y, a)
    qsig, qbk0, qbk1 = [bitor(q[slc,:]) for slc in slices]

    #subtract the background
    area_ratio = float(extrsize)/bkarea
    f1d = fsig - area_ratio*(fbk0 + fbk1)
    e1d = np.sqrt(esig**2 + (area_ratio*ebk0)**2 + (area_ratio*ebk1)**2)

    #propagate the data quality flags
    q1d = qsig | qbk0 | qbk1

    #construct wavelength array
    xref, wref, dwdx = [x2d['sci'].header[s] for s in ['crpix1','crval1','cd1_1']]
    x = np.arange(f.shape[0] + 1)
    wedges = wref + (x - xref + 0.5)*dwdx
    w0, w1 = wedges[:-1], wedges[1:]

    #construct exposure time array
    expt = np.ones(f.shape[0])*x2d['sci'].header['exptime']

    #-----PUT INTO TABLE-----
    #make data columns
    colnames = ['w0','w1','flux','error','dq','exptime']
    units = ['Angstrom']*2 + ['ergs/s/cm2/Angstrom']*2 + ['s']
    descriptions = ['left (short,blue) edge of the wavelength bin',
                    'right (long,red) edge of the wavelength bin',
                    'average flux over the bin',
                    'error on the flux',
                    'data quality flags',
                    'cumulative exposure time for the bin']
    dataset = [w0, w1, f1d, e1d, q1d, expt]
    cols = [Column(d,n,unit=u,description=dn) for d,n,u,dn in
            zip(dataset, colnames, units, descriptions)]

    #make metadata dictionary
    descriptions = {'rootname' : 'STScI identifier for the dataset used to '
                                 'create this spectrum.'}
    meta = {'descriptions' : descriptions,
            'rootname' : x2d[1].header['rootname'],
            'traceloc' : traceloc,
            'extrsize' : extrsize,
            'bkoff' : bkoff,
            'bksize' : bksize}

    #put into table
    tbl  = Table(cols, meta=meta)

    #-----PUT INTO FITS-----
    if fitsout is not None:
        #spectrum hdu
        fmts = ['E']*4 + ['I', 'E']
        cols = [fits.Column(n,fm,u,array=d) for n,fm,u,d in
                zip(colnames, fmts, units, dataset)]
        del meta['descriptions']
        spechdr = fits.Header(meta.items())
        spechdu = fits.BinTableHDU.from_columns(cols, header=spechdr,
                                                name='spectrum')

        #make primary header
        prihdr = fits.Header()
        prihdr['comment'] = ('Spectrum generated from an x2d file produced by '
                             'STScI. The dataset is identified with the header '
                             'keywrod rootname. All pixel locations refer to '
                             'the x2d and are indexed from 0. '
                             'Created with spectralPhoton software '
                             'http://github.com/parkus/spectralPhoton')
        prihdr['date'] = strftime('%c')
        prihdr['rootname'] = x2d[1].header['rootname']
        prihdu = fits.PrimaryHDU(header=prihdr)

        hdulist = fits.HDUList([prihdu,spechdu])
        hdulist.writeto(fitsout, clobber=clobber)

    return tbl

def __parsetags(tag, x1d, wr, ysignal, yback):
    #spectrify the counts and put them in an array
    spectrify(tag, x1d)
    cos = __iscos(x1d)

    #get the data from the hdus
    names = ['xdisp','time','wavelength','epera']
    if cos: names.append('epsilon')
    cntsarr = lambda t: np.array([t.data[s] for s in names])
    cntslist = [cntsarr(t) for t in tag if t.name == 'EVENTS']
    cnts = np.hstack(cntslist)

    #get rid of superfluous counts
    keep = mnp.inranges(cnts[2], wr)
    cnts = cnts[:,keep]
    if cnts.shape[1] == 0:
        return cnts[1:]

    #squish
    wtrows = [-1, -2] if cos else [-1]
    cnts = sp.squish(cnts, ysignal, yback, weightrows=wtrows)
    return cnts[1:]

def __tinfo(tag, mjdref, dt, waste):
    mjdstart = tag[1].header['expstart']
    toff = 86400.0*(mjdstart - mjdref)
    tr = [tag[2].data[k][0] for k in ['start','stop']]
    if not waste:
        dt = int(round((tr[1] - tr[0])/dt))
        if dt == 0: dt = 1
    return toff, tr, dt, mjdstart

def stsciribbons(x1d, seg=''):
    """
    Get the default ribbons that stsci used for the x1d extractions, for use
    with spectralPhoton.divvy
    """
    if x1d is not None:
        cos, stis = __iscos(x1d), __isstis(x1d)
        xh, xd = x1d[1].header, x1d[1].data

    if cos:
        seg = seg[-1]
        extrsize = xh['sp_hgt_'+seg]
        bksize = [xh['b_hgt1_'+seg], xh['b_hgt2_'+seg]]
        ytrace = xh['sp_loc_'+seg]
        ybk1, ybk2 = xh['b_bkg1_'+seg], xh['b_bkg2_'+seg]
        bkoff = [ybk1-ytrace, ybk2-ytrace]
    if stis:
        extrsize = np.median(xd['extrsize'])
        bksize = map(np.median, [xd['bk1size'], xd['bk2size']])
        bkoff = map(np.median, [xd['bk1offst'], xd['bk2offst']])

    ysignal = np.array([-1.0, 1.0])*extrsize
    if not hasattr(bkoff, '__iter__'): bkoff = [bkoff]
    if not hasattr(bksize, '__iter__'): bkoff = [bksize]

    bksize,bkoff = map(np.array, [bksize, bkoff])
    yback = np.array([bkoff - bksize, bkoff + bksize]).T

    if yback[0,1] > ysignal[0]: yback[0,1] = ysignal[0]
    if yback[1,0] < ysignal[1]: yback[1,0] = ysignal[1]

    return ysignal, yback

def __ribbons(x1d, extrsize, bksize, bkoff, seg=''):
    """Get coordinates for extraction ribbons."""
    #TODO: utilize stsci ribbons to shorten, either here or in the auto funcs
    if x1d is not None:
        cos, stis = __iscos(x1d), __isstis(x1d)
        xh, xd = x1d[1].header, x1d[1].data
    if cos: seg = seg[-1]

    if extrsize == 'stsci':
        if cos: extrsize = xh['sp_hgt_'+seg]
        if stis: extrsize = np.median(xd['extrsize'])
    ysignal = np.array([-1.0, 1.0])*extrsize

    if bksize == 'stsci':
        if cos: bksize = [xh['b_hgt1_'+seg], xh['b_hgt2_'+seg]]
        if stis: bksize = map(np.median, [xd['bk1size'], xd['bk2size']])

    if bkoff == 'stsci':
        if cos:
            ytrace = xh['sp_loc_'+seg]
            ybk1, ybk2 = xh['b_bkg1_'+seg], xh['b_bkg2_'+seg]
            bkoff = [ybk1-ytrace, ybk2-ytrace]
        if stis:
            bkoff = map(np.median, [xd['bk1offst'], xd['bk2offst']])
    if not hasattr(bkoff, '__iter__'): bkoff = [bkoff]
    if not hasattr(bksize, '__iter__'): bkoff = [bksize]

    bksize,bkoff = map(np.array, [bksize, bkoff])
    yback = np.array([bkoff - bksize, bkoff + bksize]).T

    if yback[0,1] > ysignal[0]: yback[0,1] = ysignal[0]
    if yback[1,0] < ysignal[1]: yback[1,0] = ysignal[1]

    return ysignal, yback

def __findx1d(tagfile, x1dfiles):
    x1dfile = path.basename(x1dutils.tag2x1d(tagfile))
    match = filter(lambda s: x1dfile in s, x1dfiles)
    if len(match):
        return match[0]
    else:
        raise ValueError('There is no x1d file corresponding to {} in '
                         'x1dfiles.'.format(path.basename(tagfile)))

def __findtagb(tagafile, tagbfiles):
    if 'corrtag_a' in tagafile:
        tagbfile = tagafile.replace('corrtag_a', 'corrtag_b')
        match = filter(lambda s: tagbfile in s, tagbfiles)
        if len(match):
            return match[0]
        else:
            return None