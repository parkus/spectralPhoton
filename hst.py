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


# TODO: test with COS NUV and STIS echelle data

def readtagset(directory, traceloc='stsci', fluxed='tag_vs_x1d', divvied=True, clipends=True):
    """

    Parameters
    ----------
    directory
    traceloc
    fluxed
    divvied
    clipends

    Returns
    -------
    Photons object
    """

    # find all the tag files and matching x1d files
    tagfiles, x1dfiles = obs_files(directory)

    # start by parsing photons from first observation
    photons = readtag(tagfiles[0], x1dfiles[0], traceloc, fluxed, divvied, clipends)

    # now prepend/append other observations (if available) in order of time
    if len(tagfiles) > 1:
        for tagfile, x1dfile in zip(tagfiles[1:], x1dfiles[1:]):
            photons2 = readtag(tagfile, x1dfile, traceloc, fluxed, divvied, clipends)

            # add in order of time
            if photons2.time_datum < photons.time_datum:
                photons = photons2 + photons
            else:
                photons = photons + photons2

    return photons


def readtag(tagfile, x1dfile, traceloc='stsci', fluxed='tag_vs_x1d', divvied=True, clipends=True):
    """

    Parameters
    ----------
    tagfile
    x1dfile
    traceloc
    fluxed
    divvied
    clipends

    Returns
    -------
    Photons object
    """

    # open tag file
    tag = _fits.open(tagfile)

    # is it a STIS or COS observation?
    stis = _isstis(tag)
    cos = _iscos(tag)

    if traceloc in [None, 'none', False]:
        traceloc = 0.0
    fluxit = fluxed not in ['none', None, False]
    divvyit = divvied or fluxit
    if divvyit and traceloc == 0.0:
        raise ValueError('Cannot atuomatically divvy events into signal and background regions if a trace '
                         'location is not specified or et to 0.')
    if traceloc != 'stsci' and fluxit:
        raise ValueError('Proper computation of effective area of photon wavelengths (and thus flux) requires '
                         'that traceloc==\'stsci\'.')
    if x1dfile is None:
        if fluxed or divvied:
            raise ValueError('Cannot flux or divvy the events if no x1d is available.')
        if traceloc == 'stsci':
            raise ValueError('If x1d is not provided, the STScI trace location (traceloc) is not known.')

    # open x1d file
    x1d = _fits.open(x1dfile) if x1dfile is not None else None

    # empty photons object
    photons = _sp.Photons()

    # parse observation metadata
    hdr = tag[0].header + tag[1].header
    if x1d is not None:
        hdr += x1d[1].header
    photons.obs_metadata = [hdr]

    # parse observation time datum
    photons.time_datum = _time.Time(hdr['expstart'], format='mjd')

    # parse observation time range
    gti = tag['GTI'].data
    time_ranges = _np.array([gti['start'], gti['stop']]).T
    photons.obs_times = [time_ranges]

    # parse observation wavelength ranges.
    if x1d is None:
        if stis:
            wave_ranges = _np.array([[hdr['minwave'], hdr['maxwave']]])
        if cos:
            w = tag[1].data['wavelength']
            nonzero = w > 0
            wave_ranges = _np.array([[_np.min(w[nonzero]), _np.max(w[nonzero])]])
    else:
        # if x1d is available, areas where every pixel has at least one flag matching clipflags will be
        # clipped. for STIS almost every pixel is flagged with bits 2 and 9, so these are ignored
        clipflags = 2 + 128 + 256 if stis else 8 + 128 + 256
        wave_ranges = good_waverange(x1d, clipends=clipends, clipflags=clipflags)
    photons.obs_bandpasses = [wave_ranges]

    if cos:
        # keep only the wavelength range of the appropriate segment if FUV detector
        if hdr['detector'] == 'FUV':
            i = 0 if hdr['segment'] == 'FUVA' else 1
            photons.obs_bandpasses[0] = photons.obs_bandpasses[0][[i], :]

        # parse photons. I'm going to use sneaky list comprehensions and such. sorry. this is nasty because
        # supposedly stsci sometimes puts tags into multiple 'EVENTS' extensions
        get_data = lambda extension: [extension.data[s] for s in ['time', 'wavelength', 'epsilon', 'dq', 'pha']]
        data_list = [get_data(extension) for extension in tag if extension.name == 'EVENTS']
        data = map(_np.hstack, zip(*data_list))
        photons.photons = _tbl.Table(data=data, names=['t', 'w', 'e', 'q', 'pulse_height'])

        # add cross dispersion and order info
        xdisp, order = _get_yinfo_COS(tag, x1d, traceloc)
        photons['y'], photons['o'] = xdisp, order

        # cull anomalous events
        bad_dq = 64 | 512 | 2048
        bad = (_np.bitwise_and(photons['dq'], bad_dq) > 0)
        photons.photons = photons.photons[~bad]

        # add signal/background column to photons
        if divvyit:
            if hdr['detector'] == 'NUV':
                limits = [stsci_extraction_ranges(x1d, seg) for seg in ['A', 'B', 'C']]
                ysignal, yback = zip(*limits)
                map(photons.divvy, ysignal, yback, [0, 1, 2])
            elif hdr['detector'] == 'FUV':
                seg = hdr['segment']
                ysignal, yback = stsci_extraction_ranges(x1d, seg)
                photons.divvy(ysignal, yback)

        # add effective area to photons
        if fluxit:
            if hdr['detector'] == 'FUV':
                segments = [0] if hdr['segment'] == 'FUVA' else [1]
            else:
                segments = [0, 1, 2]

            Aeff = _np.zeros_like(photons['t'])
            for i in segments:
                Aeff_i = _get_Aeff_x1d(photons, x1d, x1d_row=i, order=i, method=fluxed)
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
            # get number of orders and the order numbers
            Norders = x1d['sci'].header['naxis2']
            order_nos = x1d['sci'].data['sporder']

            Aeff = _np.zeros_like(photons['t'])
            for x1d_row, order in zip(range(Norders), order_nos):
                Aeff_i = _get_Aeff_x1d(photons, x1d, x1d_row, order, method=fluxed)
                Aeff[photons['o'] == order] = Aeff_i

            photons['a'] = Aeff

    else:
        raise NotImplementedError('HST instrument {} not recognized/code not written to handle it.'
                                  ''.format(hdr['instrume']))

    # cull photons outside of wavelength and time ranges
    keep_w = (photons['w'] >= wave_ranges[0,0]) & (photons['w'] <= wave_ranges[0,-1])
    keep_t = (photons['t'] >= time_ranges[0,0]) & (photons['t'] <= time_ranges[0,-1])
    photons.photons = photons.photons[keep_w & keep_t]

    # add appropriate units
    photons['t'].unit = _u.s
    photons['w'].unit = _u.AA
    if 'a' in photons:
        photons['a'].unit = _u.cm**2

    tag.close()
    if x1d:
        x1d.close()

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
        traceloc, extrsize, bksize, bkoff = map(intrnd, [traceloc, extrsize, bksize, bkoff])
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
    wedges = _get_x2d_waveedges(x2d)
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

    Parameters
    ----------
    tag
    x1d
    traceloc

    Returns
    -------
    xdisp, order
    """

    segment = tag[0].header['segment']
    if x1d is not None:
        xd, xh = x1d[1].data, x1d[1].header
    det = tag[0].header['detector']

    xdisp_list, order_list = [], []
    for i,t in enumerate(tag):
        if t.name != 'EVENTS': continue

        td,th = t.data, t.header



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
            if traceloc != 'stsci':
                raise NotImplementedError('NUV detector has multiple traces on the same detector, so custom traceloc '
                                          'has not been implemented.')
            segs = ['A', 'B', 'C'] # [s[-1] for s in xd['segment']] commented out to make x1d not used
            yextr = _np.array([xh['SP_LOC_' + seg] for seg in segs])
            yoff = _np.array([xh['SP_OFF_' + seg] for seg in segs])
            yspec = yextr + yoff
            xdisps = td['yfull'][_np.newaxis, :] - yspec[:, _np.newaxis]

            # associate orders with each count
            order = _np.argmin(abs(xdisps), 0)

            xdisp = xdisps[order, _np.arange(len(td['yfull']))]

        # otherwise, it's just a simple subtraction
        else:
            if traceloc == 'stsci':
                yexpected, yoff = [x1d[1].header[s+segment[-1]] for s in ['SP_LOC_','SP_OFF_']]
                yspec = yexpected + yoff
            elif traceloc == 'median':
                Npixx  = th['talen2']
                x, y = td['xfull'], td['yfull']
                yspec = _median_trace(x, y, Npixx, 8)
            elif traceloc == 'lya':
                Npixy = th['talen3']
                yspec = _lya_trace(td['wavelength'], td['yfull'], Npixy)
            elif type(traceloc) in [int, float]:
                yspec = float(traceloc)
            else:
                raise ValueError('traceloc={} not recognized.'.format(traceloc))

            xdisp = td['yfull'] - yspec
            order = _np.zeros(len(td), 'i2') if segment[-1] == 'A' else _np.ones(len(td), 'i2')

        xdisp_list.append(xdisp)
        order_list.append(order)

    xdisp, order = map(_np.hstack, [xdisp_list, order_list])

    return xdisp, order


def rectify_g140m(g140mtag):
    if type(g140mtag) is str:
        g140mtag = _fits.open(g140mtag)
    x, y = [g140mtag[1].data[s].astype('f4') for s in ['axis1', 'axis2']]

    # add some psudo-random uniform offsets between 0 and 1 pixel to avoid aliasing
    _np.random.seed(0) # for repeatability
    x += _np.random.uniform(0.0, 1.0, x.shape).astype('f4')
    y += _np.random.uniform(0.0, 1.0, y.shape).astype('f4')

    # bin to an image
    edges = _np.arange(2049) # makes 2048 bins, I verified that STScI does index the pixels from 0
    img, xe, ye = _np.histogram2d(x, y, bins=[edges]*2)

    # identify iarglow by finding maxima of each row of pixels in x direction
    x_mx = _np.argmax(img, axis=0)
    count_mx = img[x_mx,range(2048)]

    # fit a line to the airglow line, weighting by counts (amounts to weighting by S/N)
    ymids = (edges[:-1] + edges[1:])/2.0
    dxdy, x0 = _np.polyfit(ymids, x_mx, 1, w=count_mx)

    # rotate all tags to make airglow line vertical
    angle = _np.arctan(dxdy)
    c, s = _np.cos, _np.sin
    rotation_matrix = [[c(angle), -s(angle)],
                       [s(angle), c(angle)]]
    xr, yr = _np.dot(rotation_matrix, [x,y])

    # rotate x0 to get rotated coordinate of airglow center
    x0r, _ = _np.dot(rotation_matrix, [x0, 0.0])

    # use mean of events within 3 pixels (~0.08  AA) of x0 to get a better estiamte of airglow center
    # doesn't actually change things much, but whatever
    use = abs(xr - x0r) < 3
    x_airglow = _np.mean(xr[use])

    # use spectral scale in tagfile and airglow cetner to assign wavelengths to events
    w0 = 1215.67
    dwdx = g140mtag[1].header['tc2_2']
    w = w0 + (xr - x_airglow)*dwdx

    return xr,yr,w


def extract_g140m_custom(g140mtagfile, x2dfile=None, extrsize=22, bkoff=300, bksize=10):

    tag = _fits.open(g140mtagfile)

    # straighten things up so that the ariglow line is vertical and use the airglwo to calibrate wavelength
    xr, yr, w = rectify_g140m(tag)

    # now find the lya by looking for the peak redward of the airglow
    redwing = (w > 1215.8) & (w < 1216.2)
    counts, bin_eedges = _np.histogram(yr[redwing], bins=2100)
    bins_mids = (bin_eedges[:-1] + bin_eedges[1:])/2.0
    yspec = bins_mids[_np.argmax(counts)]

    # compute event distances from spectrum line
    xdisp = yr - yspec

    # make a phtons object using readtag and just replace w and y
    photons = readtag(g140mtagfile, None, traceloc=0.0, divvied=False, fluxed=False)
    photons['y'] = xdisp
    photons['w'] = w

    # divvy the photons
    ysignal = [[-extrsize/2.0, extrsize/2.0]]
    dback = _np.array([-bksize/2.0, bksize/2.0])
    yback = _np.hstack([-bkoff+dback, bkoff+dback])
    photons.divvy(ysignal, yback)

    # if x2d file present, bin tags the same as x2d and compare to get fluxes
    if x2dfile is not None:
        x2d = _fits.open(x2dfile)
        photons['a'] = _get_Aeff_x2d(photons, x2d)
        photons['a'].unit = _u.cm**2

    return photons


def _get_Aeff_x2d(photons, x2d):
        # get wavelength edges of x2d pixels
        w_bins = _get_x2d_waveedges(x2d)

        # get position of spectrum in y
        spec_pixel = x2d[1].header['crpix2']

        # create appropriate ybins centered on yspec of tags with the same number of bins above and below as x2d
        n = x2d[1].data.shape[0]
        n_below = spec_pixel
        y_below = 3 - n_below*2
        y_above = y_below + 2*n
        y_bins = _np.arange(y_below, y_above + 2, 2)

        # create an equivalently-binned image from the counts
        counts = _np.histogram2d(photons['y'], photons['w'], bins=[y_bins, w_bins])[0]

        # take ratio with x2d flux and diffuse to point correction to get effextive area within each pixel
        diff2pt, expt = [x2d[1].header[s] for s in ['diff2pt', 'exptime']]
        flux = x2d[1].data * diff2pt
        wmids = (w_bins[:-1] + w_bins[1:])/2.0
        dw = _np.diff(w_bins)
        photon_energy = _const.h*_const.c/(wmids * _u.AA)
        photon_energy = photon_energy.to(_u.erg).value
        Aeff = photon_energy[_np.newaxis, :]*counts/(flux*dw[_np.newaxis, :]*expt)

        # interpolate any values that aren't a positive real number (e.g. where the x2d flux was zero, counts were
        # zero, etc.).
        valid = _np.isfinite(Aeff) & (Aeff > 0)
        ymids = (y_bins[:-1] + y_bins[1:])/2.0
        ygrid, wgrid = _np.meshgrid(ymids, wmids)
        spline = _interp.SmoothBivariateSpline(ygrid[valid], wgrid[valid], Aeff[valid])
        Aeff[~valid] = spline(ygrid[~valid], wgrid[~valid], grid=False)

        # interpolate the effective areas
        i = _np.searchsorted(w_bins, photons['w'])
        j = _np.searchsorted(y_bins, photons['y'])

        # apparently numpy has trouble with massive indexing? I'll split this into chunkcs
        chunksize = 50000
        chunk_indices = range(0, len(photons), chunksize)
        a_list = []
        for start in chunk_indices:
            ii = i[start:start+chunksize]
            jj = j[start:start+chunksize]
            a_list.append(Aeff[ii,jj])

        return _np.hstack(a_list)


def _get_Aeff_x1d(photons, x1d, x1d_row, order, method='x1d_only'):
    """

    Parameters
    ----------
    photons
    x1d
    x1d_row
    order
    method

    Returns
    -------
    Aeff
    """

    # get x1d data
    w, cps, flux, error = [x1d[1].data[s][x1d_row] for s in ['wavelength', 'net', 'flux', 'error']]

    # estimate wavelength bin edges
    edges = _utils.wave_edges(w)

    # keep only the data in the observation ranges
    keep = (edges[:-1] >= photons.obs_bandpasses[0][0,0]) & (edges[1:] <= photons.obs_bandpasses[0][0,1])
    w, cps, flux, error = [a[keep] for a in [w, cps, flux, error]]
    edges = _np.append(edges[:-1][keep], edges[1:][keep][-1])

    # wavelength bin widths
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
        x_edges_ds = _utils.adaptive_downsample(edges, flux, error, 2.0)[0]

        # get count rate spectrum using the x1d wavelength edges
        cps_density, cps_error = photons.spectrum(edges, order=order)[2:4]

        # downsample count rate spectrum
        cps_edges_ds = _utils.adaptive_downsample(edges, cps_density, cps_error, 2.0)[0]

        # keep coarsest bins from either spectrum. this is ugly and slow, sorry
        edges_ds = [edges[0]]
        i, j = 1, 1
        while edges_ds[-1] < edges[-1]:
            if x_edges_ds[i] > cps_edges_ds[j]:
                edges_ds.append(x_edges_ds[i])
                i += 1
                while cps_edges_ds[j] < edges_ds[-1]:
                    j += 1
            else:
                edges_ds.append(cps_edges_ds[j])
                j += 1
                while x_edges_ds[i] < edges_ds[-1]:
                    i += 1
        edges_ds = _np.array(edges_ds)

        # rebin x1d flux and tag count rates to coarse (downsampled, ds) binning
        flux = _utils.rebin(edges_ds, edges, flux)
        cps_density = _utils.rebin(edges_ds, edges, cps_density)

        # I want counts for the computation below not count density
        w = (edges_ds[:-1] + edges_ds[1:])/2.0
        dw = _np.diff(edges_ds)
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

    Parameters
    ----------
    tag
    x1d
    traceloc

    Returns
    -------
    time, wave, xdisp, order, dq
    """

    is_echelle = tag[0].header['opt_elem'].startswith('E')
    if is_echelle and x1d is None:
        raise ValueError('Cannot extract events from a STIS echelle spectrum without an x1d.')

    if x1d is not None:
        xd = x1d['sci'].data

    if is_echelle and traceloc != 'stsci':
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
        if x1d is None:
            if header['TC2_3'] != 0:
                raise NotImplementedError('Whoa! I didn\'t expect that. STScI gave a nonzero value for the change in '
                                          'waveelngth with change in y pixel. Hmmm, better revise the code to deal '
                                          'with that.')
            x0, y0, dydx = [header[s] for s in ['tcrpx2', 'tcrvl2', 'tc2_2']]
            compute_wave = lambda x: (x - x0)*dydx + y0
            waveinterp = [compute_wave]
            dqinterp = [lambda x: _np.zeros(x.shape, 'uint16')]
        else:
            # number of x1d pixels
            Nx_x1d, Ny_x1d = [x1d[0].header[key] for key in ['sizaxis1','sizaxis2']]

            ## for some reason tag and x1d use different pixel scales, so get the factor of that difference
            Nx_tag, Ny_tag = header['axlen1'], header['axlen2']
            xfac, yfac = Nx_tag/Nx_x1d, Ny_tag/Ny_x1d

            ## make a vector of pixel indices
            xpix = _np.arange(1.0 + xfac/2.0, Nx_tag + 1.0, xfac)

            ## make interpolation functions
            interp = lambda vec: _interp.interp1d(xpix, vec, bounds_error=False, fill_value=_np.nan)
            extryinterp = map(interp, xd['extrlocy']*yfac)
            waveinterp = map(interp, xd['wavelength'])
            def make_dq_function(dq):
                f = _interp.interp1d(xpix, dq, 'nearest', bounds_error=False, fill_value=_np.nan)
                return lambda x: f(x).astype('uint16')
            dqinterp = map(make_dq_function, xd['dq'])


        if is_echelle:
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
            Norders = x1d['sci'].header['naxis2']
            for l in range(Norders):
                ind = (line == l)
                wave[ind] = waveinterp[l](x[ind])
                dq[ind] = dqinterp[l](x[ind])
        else:
            # interpoalte dq flags
            dq = dqinterp[0](x)

            # order is the same for all tags
            if x1d is None:
                order = _np.ones(x.shape, 'i2')
            else:
                order = xd['sporder'][0]*_np.ones(x.shape, 'i2')

            # interpolate wavelength
            wave = waveinterp[0](x)

            # get cross dispersion distance depending on specified trace location
            if type(traceloc) in [int, float]:
                yspec = traceloc
            elif traceloc == 'stsci':
                yspec = extryinterp[0](x)
            elif traceloc == 'median':
                yspec = _median_trace(x, y, Nx_tag)
            elif traceloc == 'lya':
                yspec = _lya_trace(wave, y, Ny_tag)
            else:
                raise ValueError('traceloc={} not recognized.'.format(traceloc))
            xdisp = y - yspec

        # pack the reduced data and move on to the next iteration
        data_list.append([time, wave, xdisp, order, dq])

    # unpack the data arrays and return them
    time, wave, xdisp, order, dq = map(_np.hstack, zip(*data_list))
    return time, wave, xdisp, order, dq


def good_waverange(x1d, clipends=False, clipflags=0b0000000110000100):
    """
    Returns the range of good wavelengths based on the x1d.

    clipends will clip off the areas at the ends of the spectra that have bad
    dq flags.

    Parameters
    ----------
    x1d
    clipends
    clipflags

    Returns
    -------

    """
    xd = _fits.getdata(x1d) if type(x1d) is str else x1d[1].data
    wave = xd['wavelength']
    edges = map(_utils.wave_edges, wave)
    if clipends:
        dq = xd['dq']
        minw, maxw = [], []
        for e,d in zip(edges,dq):
            dq_match = _np.bitwise_and(d, clipflags)
            good = (dq_match == 0)
            w0, w1 = e[:-1], e[1:]
            minw.append(w0[good][0])
            maxw.append(w1[good][-1])
        return _np.array([minw,maxw]).T
    else:
        return _np.array([e[[0,-1]] for e in edges])


def tagname2x1dname(tagname):
    """Determine the corresponding x1d filename from  tag filename."""
    return _re.sub('_(corr)?tag_?[ab]?.fits', '_x1d.fits', tagname)


def stsci_extraction_ranges(x1d, seg=''):
    """

    Parameters
    ----------
    x1d
    seg

    Returns
    -------
    ysignal, yback
    """
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


def _same_obs(hdus):
    rootnames = [hdu[0].header['rootname'] for hdu in hdus]
    if not all([name == rootnames[0] for name in rootnames]):
        raise Exception('The fits data units are from different observations.')


# TODO test
def _median_trace(x, y, Npix, binfac=1):
    """

    Parameters
    ----------
    x
    y
    Npix
    binfac

    Returns
    -------
    ytrace
    """

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
    return _np.polyval(p, x)


# TODO: still doesn't work in all instances. maybe mesaure wdith of lya image and select widest location?
def _lya_trace(w, y, ymax):
    """

    Parameters
    ----------
    w
    y
    ymax

    Returns
    -------
    ytrace
    """
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
    return ytrace


def obs_files(directory):
    """

    Parameters
    ----------
    directory

    Returns
    -------
    tags,x1ds
    """

    allfiles = _os.listdir(directory)
    allfiles = [_os.path.join(directory, name) for name in allfiles]
    tagfiles = filter(lambda s: 'tag' in s, allfiles)
    x1dfiles = filter(lambda s: 'x1d.fits' in s, allfiles)

    # obervation identifiers
    obsids = _np.unique([_fits.getval(f, 'rootname') for f in tagfiles])

    tags, x1ds = [],[]
    for obsid in obsids:
        # look for tag file with matching obsid in filename
        obs_tags = filter(lambda s: obsid in s, tagfiles)
        if len(obs_tags) == 0:
            raise ValueError('No tag files found for observation {}'.format(obsid))
        tags.extend(obs_tags)

        # look for x1d files with matching obsids
        obs_x1ds = filter(lambda s: obsid in s, x1dfiles)
        if len(obs_x1ds) == 0:
            raise ValueError('No x1d file found for observation {}'.format(obsid))
        if len(obs_x1ds) > 1:
            raise ValueError('Multiple x1d files found for observation {}'.format(obsid))

        # make sure to add an x1d file entry for every tag file (since the corrtag_a and corrtag_b files of cos are
        # both associated with a single x1d)
        x1ds.extend(obs_x1ds*len(obs_tags))

    return tags,x1ds


def _argsegment(x1d, segment):
    return  _np.nonzero(x1d['segment'] == segment)[0]

# TODO: check this using GJ 581 muscles i data
def _get_x2d_waveedges(x2d):
    xref, wref, dwdx = [x2d['sci'].header[s] for s in ['crpix1', 'crval1', 'cd1_1']]
    x = _np.arange(x2d[1].data.shape[0] + 1)
    wedges = wref + (x - xref + 0.5) * dwdx
    return wedges


_iscos = lambda hdu: hdu[0].header['instrume'] == 'COS'
_isstis = lambda hdu: hdu[0].header['instrume'] == 'STIS'