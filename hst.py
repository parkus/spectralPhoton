from astropy.io import fits as _fits
import astropy.time as _time
import astropy.units as _u
import astropy.constants as _const
import astropy.table as _tbl
from time import strftime as _strftime
import numpy as _np
import scipy.interpolate as _interp
import os as _os
import re as _re
import __init__ as _sp
import utils as _utils


def readtagset(directory, traceloc='stsci', fluxed='tag_vs_x1d', divvied=True, clipends=True):

    # find all the tag files and matching x1d files
    tagfiles, x1dfiles = _obs_files(directory)

    # start by parsing photons from first observation
    photons = readtag(tagfiles[0], x1dfiles[0], traceloc, fluxed, divvied, clipends)

    # now prepend/append other observations (if available) in order of time
    if len(tagfiles) > 1:
        for tagfile, x1dfile in zip(tagfiles, x1dfiles):
            photons2 = readtag(tagfile, x1dfile, traceloc, fluxed, divvied, clipends)

            # add in order of time
            if photons2.time_datum < photons.time_datum:
                photons = photons2 + photons
            else:
                photons = photons + photons2

    return photons


def readtag(tagfile, x1dfile, traceloc='stsci', fluxed='tag_vs_x1d', divvied=True, clipends=True):

    fluxit = fluxed not in ['none', None, False]
    divvyit = divvied or fluxit

    # open files
    tag, x1d = map(_fits.open, [tagfile, x1dfile])

    # is it a STIS or COS observation?
    stis = _isstis(x1d)
    cos = _iscos(x1d)

    # empty photons object
    photons = _sp.Photons()

    # parse observation metadata
    hdr = tag[0].header + tag[1].header + x1d[1].header
    photons.obs_metadata = [hdr]

    # parse observation time datum
    photons.time_datum = _time.Time(hdr['expstart'], format='mjd')

    # parse observation time range
    gti = tag['GTI'].data
    time_ranges = _np.array([gti['start'], gti['stop']]).T
    photons.obs_times = [time_ranges]

    # parse observation wavelength ranges. areas where every pixel has at least one flag matching clipflags will be
    # clipped. for STIS almost every pixel is flagged with bits 2 and 9, so these are ignored
    clipflags = 2 + 128 + 256 if stis else 8 + 128 + 256
    wave_ranges = good_waverange(x1d, clipends=clipends, clipflags=clipflags)
    photons.obs_bandpasses = [wave_ranges]

    if cos:

        # parse photons. I'm going to use sneaky list comprehensions and such. sorry. this is nasty because
        # supposedly stsci sometimes puts tags into multiple 'EVENTS' extensions
        get_data = lambda extension: [extension.data[s] for s in ['time', 'wavelength', 'epsilon', 'dq', 'pha']]
        data_list = [get_data(extension) for extension in tag if extension.name == 'EVENTS']
        data = map(_np.hstack, zip(*data_list))
        photons.ph = _tbl.Table(data=data, names=['t', 'w', 'e', 'q', 'pulse_height'])

        # add cross dispersion and order info
        xdisp, order = _get_yinfo_COS(tag, x1d, traceloc)
        photons['y'], photons['o'] = xdisp, order

        # add signal/background column to photons
        if divvyit:
            if hdr['detector'] == 'NUV':
                limits = [stsci_extraction_ranges(x1d, seg) for seg in ['A', 'B', 'C']]
                ysignal, yback = zip(limits)
                ysignal = _np.array(ysignal)
                yback = _np.vstack(yback)
            elif hdr['detector'] == 'FUV':
                seg = hdr['segment']
                ysignal, yback = stsci_extraction_ranges(x1d, seg)
            photons.divvy(ysignal, yback)

        # add effective area to photons
        if fluxit:
            if traceloc != 'stsci':
                raise ValueError('Proper computation of effective area of photon wavelengths (and thus flux) requires '
                                 'that traceloc==\'stsci\'.')
            if hdr['detector'] == 'FUV':
                segments = [0] if hdr['segment'] == 'FUVA' else [1]
            else:
                segments = [0, 1, 2]

            Aeff = _np.zeros_like(photons['t'])
            for i in segments:
                Aeff_i = _get_Aeff(photons, x1d, x1d_row=i, order=i, method=fluxed)
                Aeff[photons['o'] == i] = Aeff_i

            photons['a'] = Aeff

    elif stis:

        # nothing comes for free with STIS
        time, wave, xdisp, order, dq = _get_photon_info_STIS(tag, x1d, traceloc)
        photons.photons = _tbl.Table([time, wave, xdisp, order, dq], names=['t', 'w', 'y', 'o', 'q'])

        # add signal/background column to photons
        if divvyit:
            ysignal, yback = stsci_extraction_ranges(x1d)
            photons.divvy(ysignal, yback)

        # add effective area to photons
        if fluxit:
            if traceloc != 'stsci':
                raise ValueError('Proper computation of effective area of photon wavelengths (and thus flux) requires '
                                 'that traceloc==\'stsci\'.')

            # get number of orders and the order numbers
            Norders = x1d['sci'].header['naxis2']
            order_nos = x1d['sci'].data['sporder']

            Aeff = _np.zeros_like(photons['t'])
            for x1d_row, order in zip(range(Norders), order_nos):
                Aeff_i = _get_Aeff(photons, x1d, x1d_row, order, method=fluxed)
                Aeff[photons['o'] == order] = Aeff_i

            photons['a'] = Aeff

    else:
        raise NotImplementedError('HST instrument {} not recognized/code not written to handle it.'
                                  ''.format(hdr['instrume']))

    # cull photons outside of wavelength and time ranges
    keep_w = _sp._inbins(wave_ranges, photons['w'])
    keep_t = _sp._inbins(time_ranges, photons['t'])
    photons.photons = photons.photons[keep_w & keep_t]

    # add appropriate units
    photons['t'].unit = _u.s
    photons['w'].unit = _u.AA
    if 'a' in photons:
        photons['a'].unit = _u.cm**2

    return photons


def x2dspec(x2dfile, traceloc='max', extrsize='stsci', bksize='stsci', bkoff='stsci', x1dfile=None, fitsout=None,
            clobber=True, bkmask=0):
    """
    Creates a spectrum using the x2d file.

    Parameters
    ----------
    x2dfile : str
        Path of the x2d file.
    traceloc : {int|'max'|'lya'}, optional
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
    x2d = _fits.open(x2dfile)

    # get the flux and error from the x2d
    f, e, q = x2d['sci'].data, x2d['err'].data, x2d['dq'].data

    inst = x2d[0].header['instrume']
    if inst != 'STIS':
        raise NotImplementedError("This function cannot handle {} data at "
                                  "present.".format(inst))

    # make sure x1d is available if 'stsci' is specified for anything
    if 'stsci' in [traceloc, extrsize, bksize, bkoff]:
        try:
            x1d = _fits.open(x1dfile)
            xd = x1d[1].data
        except:
            raise ValueError("An open x1d file is needed if 'stsci' is "
                             "specified for any of the keywords.")

    # get the ribbon values
    if extrsize == 'stsci': extrsize = _np.mean(xd['extrsize'])
    if bksize == 'stsci': bksize = _np.mean([xd['bk1size'], xd['bk2size']])
    if bkoff == 'stsci':
        bkoff = _np.mean(_np.abs([xd['bk1offst'], xd['bk2offst']]))

    # select the trace location
    if traceloc == 'max':
        sn = f / e
        sn[q > 0] = 0.0
        sn[e <= 0.0] = 0.0
        maxpixel = _np.nanargmax(sn)
        traceloc = _np.unravel_index(maxpixel, f.shape)[0]
    if traceloc == 'lya':
        xmx = _np.nanmedian(_np.argmax(f, 1))
        redsum = _np.nansum(f[:, xmx+4:xmx+14], 1)
        smoothsum = _sp._smooth_sum(redsum, extrsize)/float(extrsize)
        traceloc = _np.argmax(smoothsum) + extrsize/2

    # convert everything to integers so we can make slices
    try:
        intrnd = lambda x: int(round(x))
        traceloc, extrsize, bksize, bkoff = map(intrnd, [traceloc, extrsize,
                                                         bksize, bkoff])
    except ValueError:
        raise ValueError("Invalid input for either traceloc, extrsize, bksize, "
                         "or bkoff. See docstring.")

    # convert intensity to flux
    fluxfac = x2d['sci'].header['diff2pt']
    f, e = f * fluxfac, e * fluxfac

    # get slices for the ribbons
    sigslice = slice(traceloc - extrsize // 2, traceloc + extrsize // 2 + 1)
    bk0slice = slice(traceloc - bkoff - bksize // 2, traceloc - bkoff + bksize // 2 + 1)
    bk1slice = slice(traceloc + bkoff - bksize // 2, traceloc + bkoff + bksize // 2 + 1)
    slices = [sigslice, bk0slice, bk1slice]

    # mask bad values in background regions
    if bkmask:
        badpix = (q & bkmask) > 0
        badpix[sigslice] = False  # but don't modify the signal region
        f[badpix], e[badpix], q[badpix] = 0.0, 0.0, 0
        # make a background area vector to account for masked pixels
        goodpix = ~badpix
        bkareas = [_np.sum(goodpix[slc, :], 0) for slc in slices[1:]]
        bkarea = sum(bkareas)
    else:
        bkarea = bksize * 2

    # sum fluxes in each ribbon
    fsig, fbk0, fbk1 = [_np.sum(f[slc, :], 0) for slc in slices]

    # sum errors in each ribbon
    esig, ebk0, ebk1 = [_np.sqrt(_np.sum(e[slc, :]**2, 0)) for slc in slices]

    # condense dq flags in each ribbon
    bitor = lambda a: reduce(lambda x, y: x | y, a)
    qsig, qbk0, qbk1 = [bitor(q[slc, :]) for slc in slices]

    # subtract the background
    area_ratio = float(extrsize) / bkarea
    f1d = fsig - area_ratio * (fbk0 + fbk1)
    e1d = _np.sqrt(esig**2 + (area_ratio * ebk0)**2 + (area_ratio * ebk1)**2)

    # propagate the data quality flags
    q1d = qsig | qbk0 | qbk1

    # construct wavelength array
    xref, wref, dwdx = [x2d['sci'].header[s] for s in ['crpix1', 'crval1', 'cd1_1']]
    x = _np.arange(f.shape[0] + 1)
    wedges = wref + (x - xref + 0.5) * dwdx
    w0, w1 = wedges[:-1], wedges[1:]

    # construct exposure time array
    expt = _np.ones(f.shape[0]) * x2d['sci'].header['exptime']

    # -----PUT INTO TABLE-----
    # make data columns
    colnames = ['w0', 'w1', 'flux', 'error', 'dq', 'exptime']
    units = ['Angstrom'] * 2 + ['ergs/s/cm2/Angstrom'] * 2 + ['s']
    descriptions = ['left (short,blue) edge of the wavelength bin',
                    'right (long,red) edge of the wavelength bin',
                    'average flux over the bin',
                    'error on the flux',
                    'data quality flags',
                    'cumulative exposure time for the bin']
    dataset = [w0, w1, f1d, e1d, q1d, expt]
    cols = [_tbl.Column(d, n, unit=u, description=dn) for d, n, u, dn in
            zip(dataset, colnames, units, descriptions)]

    # make metadata dictionary
    descriptions = {'rootname': 'STScI identifier for the dataset used to '
                                'create this spectrum.'}
    meta = {'descriptions': descriptions,
            'rootname': x2d[1].header['rootname'],
            'traceloc': traceloc,
            'extrsize': extrsize,
            'bkoff': bkoff,
            'bksize': bksize}

    # put into table
    tbl = _tbl.Table(cols, meta=meta)

    # -----PUT INTO FITS-----
    if fitsout is not None:
        # spectrum hdu
        fmts = ['E'] * 4 + ['I', 'E']
        cols = [_fits.Column(n, fm, u, array=d) for n, fm, u, d in
                zip(colnames, fmts, units, dataset)]
        del meta['descriptions']
        spechdr = _fits.Header(meta.items())
        spechdu = _fits.BinTableHDU.from_columns(cols, header=spechdr,
                                                name='spectrum')

        # make primary header
        prihdr = _fits.Header()
        prihdr['comment'] = ('Spectrum generated from an x2d file produced by '
                             'STScI. The dataset is identified with the header '
                             'keywrod rootname. All pixel locations refer to '
                             'the x2d and are indexed from 0. '
                             'Created with spectralPhoton software '
                             'http://github.com/parkus/spectralPhoton')
        prihdr['date'] = _strftime('%c')
        prihdr['rootname'] = x2d[1].header['rootname']
        prihdu = _fits.PrimaryHDU(header=prihdr)

        hdulist = _fits.HDUList([prihdu, spechdu])
        hdulist.writeto(fitsout, clobber=clobber)

    return tbl


def _get_yinfo_COS(tag, x1d, traceloc='stsci'):
    """
    Add spectral units (wavelength, cross dispersion distance, energy/area)
    to the photon table in the fits data unit "tag".

    For G230L, you will get several 'xdisp' columns -- one for each segment. This allows for the use of overlapping
    background regions.
    """
    segment = tag[0].header['segment']
    xd, xh = x1d[1].data, x1d[1].header
    det = tag[0].header['detector']

    xdisp_list, order_list = [], []
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue

        td,th = t.data, t.header

        # for FUV detector "order" will be used to specify A or B segment
        if det == 'FUV':
            n = len(td['time'])
            order = _np.zeros(n) if segment[-1] == 'A' else _np.ones(n)

        if traceloc != 'stsci' and det == 'NUV':
            raise NotImplementedError('NUV detector has multiple traces on the same detector, so custom traceloc '
                                      'has not been implemented.')
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

            # all "orders" (segments) of the NUV spectra fall on the same detector and are just offset in y,
            # so find the distance of the counts from each order to find which one they match with
            if det == 'NUV':
                segs = [s[-1] for s in xd['segment']]
                yextr = _np.array([xh['SP_LOC_' + seg] for seg in segs])
                yoff = _np.array([xh['SP_OFF_' + seg] for seg in segs])
                yspec = yextr + yoff
                xdisps = td['yfull'][_np.newaxis, :] - yspec[:, _np.newaxis]

                # associate orders with each count
                order = _np.argmin(abs(xdisps), 0)

                xdisp = xdisps[order, _np.arange(len(td['yfull']))]
            # otherwise, it's just a simple subtraction
            else:
                yexpected, yoff = [x1d[1].header[s+segment[-1]] for s in ['SP_LOC_','SP_OFF_']]
                yspec = yexpected + yoff
                xdisp = td['yfull'] - yspec
                order = _np.zeros_like(xdisp) if segment[-1] == 'A' else _np.ones_like(xdisp)

        if traceloc == 'median':
            Npixx  = th['talen2']
            x, y = td['xfull'], td['yfull']
            xdisp = _median_trace(x, y, Npixx, 8)

        if traceloc == 'lya':
            Npixy = th['talen3']
            xdisp = _lya_trace(td['wavelength'], td['yfull'], Npixy)

        xdisp_list.append(xdisp)
        order_list.append(order)

    xdisp, order = map(_np.hstack, [xdisp_list, order_list])

    return xdisp, order


def _get_Aeff(photons, x1d, x1d_row, order, method='x1d_only'):

    # get x1d data
    w, cps, flux, error = [x1d[1].data[s][x1d_row] for s in ['wavelength', 'net', 'flux', 'error']]

    # estimate wavlength bin edges and widths
    edges = _utils.wave_edges(w)
    dw = _np.diff(edges)

    if method == 'x1d_only':
        # compare net count rate to flux to compute effective area vs pixel, ignoring zero-flux pixels
        keep = (cps != 0)
        flux, cps, dw, w = [v[keep] for v in [flux, cps, dw, w]]

    elif method == 'tag_vs_x1d':

        if 'r' not in photons:
            raise ValueError('Photons must have region information (signal/background) for tag_vs_x1d fluxing.')

        # downsample the x1d until bins all have a S/N of at least 2. otherwise for low-signal spectra there will be
        # many bins where the effective area is computed to be negative
        edges, flux, error = _utils.adaptive_downsample(edges, flux, error, 2.0)

        # get count rate spectrum using the x1d wavelength edges
        cps_density = photons.spectrum(edges, order=order)[2]
        w = (edges[:-1] + edges[1:])/2.0
        dw = _np.diff(edges)
        cps = cps_density*dw

    else:
        raise ValueError('fluxmethod not recognized.')

    # compare count rate to x1d flux to compute effective area grid
    avg_energy = _const.h*_const.c / (w * _u.AA)  # not quite right but fine for dw/w << 1
    avg_energy = avg_energy.to('erg').value
    Aeff_grid =  cps*avg_energy/(dw*flux)

    # interpolate the effective areas at the photon wavelengths
    in_order = (photons['o'] == order)
    Aeff = _np.interp(photons['w'][in_order], w, Aeff_grid)

    return Aeff


def _get_photon_info_STIS(tag, x1d, traceloc='stsci'):
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
        raise NotImplemented('Cannot manually determine the spectral trace locations on an echellogram.')

    data_list = [] # I will pack the data arrays pulled/computed from each 'events' extension into this
    for i,extension in enumerate(tag):
        if extension.name != 'EVENTS': continue
        events, header = extension.data, extension.header

        # get time in s (stsci uses some arbitrary scale)
        # uh, apparently they stopped doing this for some pipeline revision, so better check if it's necessary
        time = events['time']
        tratio_unscaled = time[-1]/tag['gti'].data['stop'][-1]
        tratio_scaled = tratio_unscaled * header['tscal1']
        if abs(tratio_scaled - 1.0) < abs(tratio_scaled - 1.0):
            time *= header['tscal1']

        x,y = events['axis1'],events['axis2']
        # there seem to be issues with at the stsci end with odd and even pixels having systematically different
        # values (at least for g230l) so group them by 2-pixel
        xeven, yeven = (x % 2 == 1), (y % 2 == 1)
        x[xeven] = x[xeven] - 1
        y[yeven] = y[yeven] - 1

        # add random offsets within pixel range to avoid wavelength aliasing issues from quantization
        _np.random.seed(0) # for reproducibility
        x = x + _np.random.random(x.shape)*2.0
        y = y + _np.random.random(y.shape)*2.0

        # compute interpolation functions for the dispersion line y-value and the wavelength solution for each order
        ## for some reason tag and x1d use different pixel scales, so get the factor of that difference
        Nx_tag, Ny_tag = header['axlen1'], header['axlen2']
        xfac, yfac = Nx_tag/Nx_x1d, Ny_tag/Ny_x1d

        ## make a vector of pixel indices
        xpix = _np.arange(1.0 + xfac/2.0, Nx_tag + 1.0, xfac)

        ## now make interpolation functions, one for each order
        interp = lambda vec: _interp.interp1d(xpix, vec, bounds_error=False, fill_value=_np.nan)
        extryinterp = map(interp, xd['extrlocy']*yfac)
        waveinterp = map(interp, xd['wavelength'])
        dqinterp = [_interp.interp1d(xpix, dq, 'nearest', bounds_error=False, fill_value=_np.nan) for dq in xd['dq']]

        if Norders > 1:
            # associate each tag with an order by choosing the closest order. I am using line to count the orders
            # from zero whereas order gives the actual spectral order on the Echelle
            xdisp = _np.array([y - yint(x) for yint in extryinterp])
            line = _np.argmin(abs(xdisp), 0)

            # now get the cross dispersion distance and order for each tag
            xdisp = xdisp[line, _np.arange(len(x))]
            order = xd['sporder'][line]

            # and interpolate the wavelength and data quality flags
            # looping through orders is 20x faster than looping through tags
            wave, dq = _np.zeros(x.shape), _np.zeros(x.shape, int)
            for l in range(Norders):
                ind = (line == l)
                wave[ind] = waveinterp[l](x[ind])
                dq[ind] = dqinterp[l](x[ind])
        else:
            # interpoalte dq flags
            dq = dqinterp[0](x)

            # order is the same for all tags
            order = xd['sporder'][0]*_np.ones(x.shape)

            # interpolate wavelength
            wave = waveinterp[0](x)

            # get cross dispersion distance depending on specified trace location
            if type(traceloc) in [int, float]:
                xdisp = (y - traceloc)
            elif traceloc == 'stsci':
                xdisp = (y - extryinterp[0](x))
            elif traceloc == 'median':
                xdisp = _median_trace(x, y, Nx_tag)
            elif traceloc == 'lya':
                xdisp = _lya_trace(wave, y, Ny_tag)
            else:
                raise ValueError('Traceloc value of {} not recognized.'.format(traceloc))

        # pack the reduced data and move on to the next iteration
        data_list.append([time, wave, xdisp, order, dq])

    # unpack the data arrays and return them
    time, wave, xdisp, order, dq = map(_np.hstack, zip(*data_list))
    return time, wave, xdisp, order, dq


def good_waverange(x1d, clipends=False, clipflags=0b0000000110000100):
    """Returns the range of good wavelengths based on the x1d.

    clipends will clip off the areas at the ends of the spectra that have bad
    dq flags.
    """
    xd = _fits.getdata(x1d) if type(x1d) is str else x1d[1].data
    wave = xd['wavelength']
    if clipends:
        dq = xd['dq']
        minw, maxw = [], []
        for w,d in zip(wave,dq):
            dq_match = _np.bitwise_and(d, clipflags)
            good = (dq_match == 0)
            minw.append(w[good][0])
            maxw.append(w[good][-1])
        return _np.array([minw,maxw]).T
    else:
        return _np.array([w[[0,-1]] for w in wave])


def tagname2x1dname(tagname):
    """Determine the corresponding x1d filename from  tag filename."""
    return _re.sub('_(corr)?tag_?[ab]?.fits', '_x1d.fits', tagname)


def stsci_extraction_ranges(x1d, seg=''):
    cos, stis = _iscos(x1d), _isstis(x1d)
    xh, xd = x1d[1].header, x1d[1].data

    # below these will all be divided by 2 (except bk off). initially they specify the full size
    if cos:
        seg = seg[-1]
        extrsize = xh['sp_hgt_' + seg]
        bksize = _np.array([xh['b_hgt1_' + seg], xh['b_hgt2_' + seg]])
        ytrace = xh['sp_loc_' + seg]
        ybk1, ybk2 = xh['b_bkg1_' + seg], xh['b_bkg2_' + seg]
        bkoff = [ybk1 - ytrace, ybk2 - ytrace]

    if stis:
        extrsize = _np.median(xd['extrsize'])
        bksize = map(_np.median, [xd['bk1size'], xd['bk2size']])
        bkoff = map(_np.median, [xd['bk1offst'], xd['bk2offst']])

    ysignal = _np.array([-0.5, 0.5]) * extrsize

    if not hasattr(bkoff, '__iter__'): bkoff = [bkoff]
    if not hasattr(bksize, '__iter__'): bksize = [bksize]
    bksize, bkoff = map(_np.array, [bksize, bkoff])
    yback = _np.array([bkoff - bksize/2.0, bkoff + bksize/2.0]).T

    # make sure there is no overlap between the signal and background regions
    if bkoff is 'stsci':
        if yback[0, 1] > ysignal[0]: yback[0, 1] = ysignal[0]
        if yback[1, 0] < ysignal[1]: yback[1, 0] = ysignal[1]

    return ysignal, yback


def _x1d_Aeff_solution(x1d):
    """Uses the x1d file to create a function that computes the energy/area
    [erg/cm**2] for a count of a given wavelength and spectral line (row no. in
    the x1d arrays).
    """
    #get epera from x1d
    if type(x1d) is str: x1d = _fits.open(x1d)
    wave, cps, flux = [x1d[1].data[s] for s in ['wavelength', 'net', 'flux']]
    dwave = _np.zeros(wave.shape)
    dwave[:,:-1] = wave[:,1:] - wave[:,:-1]
    dwave[:,-1] = dwave[:,-2]
    flux, cps, dwave, wave = map(list, [flux, cps, dwave, wave])
    for i in range(len(flux)):
        keep = (cps[i] != 0)
        flux[i], cps[i], dwave[i], wave[i] = [v[keep] for v in
                                              [flux[i],cps[i],dwave[i], wave[i]]]
    EperAperCount = [f/c*d for f,c,d in zip(flux,cps,dwave)]

    #make an inerpolation function for each order
    intps = [_np.interp1d(w,E,bounds_error=False) for w,E in zip(wave,EperAperCount)]

    #the function to be returned. it chooses the proper interpolation function
    #based on the orders of the input counts, then uses it to get epera from
    #wavelength
    def f(waves, orders=0):
        if type(orders) == int:
            eperas = intps[orders](waves)
        else:
            eperas = _np.zeros(waves.shape)
            for i in range(len(intps)):
                pts = (orders == i)
                eperas[pts] = intps[i](waves[pts])
        return eperas

    return f


def _same_obs(hdus):
    rootnames = [hdu[0].header['rootname'] for hdu in hdus]
    if not all([name == rootnames[0] for name in rootnames]):
        raise Exception('The fits data units are from different observations.')


def _median_trace(x, y, Npix, binfac=1):
    # NOTE: I looked into trying to exclude counts during times when the
    # calibration lamp was on for COS, but this was not easily possible as of
    # 2014-11-20 because the lamp flashes intermittently and the times aren't
    # recorded in the corrtag files

    # get the median y value and rough error in each x pixel
    bins = _np.arange(0,Npix+1, binfac)
    bin_no = _np.searchsorted(bins, x)
    binned = [y[bin_no == i] for i in xrange(1, len(bins))]
    meds = _np.array(map(_np.median, binned))
    sig2 = _np.array(map(_np.var, binned))
    Ns = _np.array(map(len, binned))
    sig2[Ns <= 1] = _np.inf
    ws = Ns/sig2

    # fit a line and subtrqact it from the y values
    midpts = (bins[:-1] + bins[1:])/2.0
    p = _np.polyfit(midpts, meds, 1, w=ws)
    return y - _np.polyval(p, x)


def _lya_trace(w, y, ymax):
    lya_range = [1214.5,1217.2]
    in_lya = (w > lya_range[0]) & (w < lya_range[1])
    y = y[in_lya]
    ytrace_old = _np.inf
    ytrace = _np.median(y)
    dy = ymax/2.0
    #iterative narrow down the yrange to block out the distorted airglow
    while abs(ytrace - ytrace_old)/ytrace > 1e-4:
        dy *= 0.5
        in_region = (y > ytrace-dy) & (y < ytrace+dy)
        y = y[in_region]
        ytrace_old = ytrace
        ytrace = _np.median(y)
    return y - ytrace


def _obs_files(directory):

    allfiles = _os.listdir(directory)
    tagfiles = filter(lambda s: 'tag' in s, allfiles)
    x1dfiles = filter(lambda s: 'x1d.fits' in s, allfiles)

    # obervation identifiers
    obsids = _np.unique([f[:9] for f in tagfiles])

    tags, x1ds = [],[]
    for obsid in obsids:
        # look for tag file with matching obsid in filename
        tags = filter(lambda s: obsid in s, tagfiles)
        if len(tags) == 0:
            raise ValueError('No tag files found for observation {}'.format(obsid))
        tags.extend([_os.path.join(directory, tag) for tag in tags])

        # look for x1d files with matching obsids
        x1ds = filter(lambda s: obsid in s, x1dfiles)
        if len(x1ds) == 0:
            raise ValueError('No x1d file found for observation {}'.format(obsid))
        if len(x1ds) > 1:
            raise ValueError('Multiple x1d files found for observation {}'.format(obsid))

        # make sure to add an x1d file entry for every tag file (since the corrtag_a and corrtag_b files of cos are
        # both associated with a single x1d)
        x1ds.extend([_os.path.join(directory, x1ds[0])]*len(tags))

    return tags,x1ds


def _argsegment(x1d, segment):
    return  _np.nonzero(x1d['segment'] == segment)[0]


_iscos = lambda x1d: x1d[0].header['instrume'] == 'COS'
_isstis = lambda x1d: x1d[0].header['instrume'] == 'STIS'