import astropy.io.fits as _fits
import astropy.time as _time
import astropy.table as _tbl
import astropy.units as _u
import astropy.constants as _const
import numpy as _np

# import mypy.my_numpy as mynp


_format_dict = {'int8':'I', 'int16':'I', 'int32':'J', 'int64':'K', 'float32':'E', 'float64':'D'}
_name_dict = {'t':'time', 'w':'wavelength', 'y':'xdisp', 'a':'area_eff', 'q':'qualflag', 'o':'order', 'n':'obs_no',
              'e':'wght', 'r':'rgn_wght'}


class Photons:
    """
    Base class for holding spectral photon data. Each instance contains a list of observed photons, including,
    at a minimum:
    - time of arrival, t
    - wavelength, w

    Optionally, the list may also include:
    - detector effective area at arrival location, a
    - data quality flags, q
    - spectrum order number, o
    - cross dispersion distance from spectral trace, y
    - event weight (termed epsilon for HST observations), e
    - region weight (1.0 for signal, -signal_area/backgroun_area for background, 0.0 otherwsise), r
    - anything else the user want's to define

    Some operators have been defined for this class:
    #FIXME

    These attributes are all protected in this class and are accessed using the bracket syntax, e.g. if `photons` is an
    instance of class `Photons`, then

    >>> photons['w']

    will return the photon wavelengths.

    I thought about deriving this class from FITS_rec or _np.recarray, but ultimately it felt cumbersome. I decided
    it's better they be a part of the object, rather than a parent to it, since it will also contain various metadata.
    """

    # for ease of use, map some alternative names to the proper photon property names
    _alternate_names = {'time':'t',
                        'wavelength':'w', 'wave':'w', 'wvln':'w', 'wav':'w', 'waveln':'w',
                        'effective area':'a', 'area':'a', 'area_eff':'a',
                        'data quality':'q', 'dq':'q', 'quality':'q', 'flags':'q', 'qualflag':'q',
                        'order':'o', 'segment':'o', 'seg':'o',
                        'observation':'n', 'obs':'n','obs_no':'n',
                        'xdisp':'y', 'cross dispersion':'y',
                        'weight':'e', 'eps':'e', 'epsilon':'e', 'event weight':'e', 'wt':'e', 'wght':'e',
                        'region':'r', 'ribbon':'r', 'rgn_wght':'r'}

    # ---------
    # OBJECT HANDLING

    def __init__(self, **kwargs):
        """
        Create an empty Photons object. This is really just to show what the derived classes must define. If I ever
        see a need to be able to create Photons objects using arrays of data in the future, then I'll rewrite this.

        Parameters
        ----------
        obs_metadata : list of metadata dict (or similar types) for each obs
        time_datum : time object
        data : astropy Table with at least a 't' and 'w' column (see class description for what to name others

        Returns
        -------
        A Photons object.
        """

        # metadata associated with the observations that recorded the photons, one list entry for each observation
        self.obs_metadata = kwargs.get('obs_metadata', [{}])

        self.time_datum = kwargs.get('time_datum', _time.Time('2000-01-01T00:00:00'))

        if 'data' in kwargs:
            self.photons = kwargs['data']
        else:
            self.photons = _tbl.Table(names=['t', 'w'], dtype=['f8', 'f8'])
            self.photons['t'].unit = _u.s
            self.photons['w'].unit = _u.AA

        if 'obs_times' in kwargs:
            self.obs_times = kwargs['obs_times']
        else:
            if 'n' in self and len(self) > 0:
                self.photons.group_by('n')
                rngs = [_np.array([[a['t'].min(), a['t'].max()]]) for a in self.photons.groups]
                self.obs_times = rngs
            elif len(self) > 0:
                self.obs_times = [_np.array([[self.photons['t'].min(), self.photons['t'].max]])]
            else:
                self.obs_times = [_np.array([[]])]

        if 'obs_bandpasses' in kwargs:
            self.obs_bandpasses = kwargs['obs_bandpasses']
        else:
            if 'n' in self and len(self) > 0:
                self.photons.group_by('n')
                rngs = [_np.array([[a['w'].min(), a['w'].max()]]) for a in self.photons.groups]
                self.obs_bandpasses = rngs
            elif len(self) > 0:
                self.obs_bandpasses = [_np.array([[self.photons['w'].min(), self.photons['w'].max]])]
            else:
                self.obs_bandpasses = [_np.array([[]])]


    def __getitem__(self, key):
        key = self._get_proper_key(key)
        return self.photons[key]


    def __setitem__(self, key, value):
        try:
            key = self._get_proper_key(key)
            self.photons[key] = value
        except KeyError:
            # try to make a new column. this might be dangerous
            if _np.isscalar(value):
                value = [value]*len(self)
            col = _tbl.Column(data=value, name=key)
            self.photons.add_column(col)


    def __len__(self):
        return len(self.photons)


    def __add__(self, other):
        """
        When adding, the photon recarrays will be added. The observation numbers will be adjusted or added as
        appropriate and times will be referenced to the first of the two objects.

        Parameters
        ----------
        other

        Returns
        -------

        """

        if not isinstance(other, Photons):
            raise ValueError('Can only add a Photons object to another Photons object.')

        # don't want to modify what is being added
        other = other.copy()

        # make column units consistent with self
        other.match_units(self)

        # add and /or update observation columns as necessary
        self.add_observations_column()
        other.add_observations_column()
        n_obs_self = len(self.obs_metadata)
        other['n'] += n_obs_self

        # re-reference times to the datum of self
        other.set_time_datum(self.time_datum)

        # stack the data tables
        photons = _tbl.vstack([self, other])

        # leave it to the user to deal with sorting and grouping and dealing with overlap as they see fit :)

        obs_metadata = self.obs_metadata + other.obs_metadata
        obs_times = self.obs_times + other.obs_times
        obs_bandpasses = self.obs_bandpasses + other.obs_bandpasses

        return Photons(photons=photons, obs_metadata=obs_metadata, time_datum=self.time_datum, obs_times=obs_times,
                       obs_bandpasses=obs_bandpasses)


    def copy(self):
        new = Photons()
        new.obs_metadata = [item.copy() for item in self.obs_metadata]
        new.time_datum = self.time_datum
        new.obs_times = [item.copy() for item in self.obs_times]
        new.photons = self.photons.copy()
        new.obs_bandpasses = [item.copy() for item in self.obs_bandpasses]
        return new


    def __contains__(self, item):
        return item in self.photons.colnames


    def writeFITS(self, path, overwrite=False):

        primary_hdu = _fits.PrimaryHDU()

        # save photontable to first extension
        photon_cols = []
        for colname in self.photons.colnames:
            tbl_col = self['colname']
            name = _name_dict[colname]
            format = _format_dict[tbl_col.dtype]
            fits_col = _fits.Column(name=name, format=format, array=tbl_col.data, unit=str(tbl_col.unit))
            photon_cols.append(fits_col)
        photon_hdr = _fits.Header()
        photon_hdr['zerotime'] = (self.time_datum.jd, 'julian date of time = 0.0')
        photon_hdu = _fits.BinTableHDU.from_columns(photon_cols, header=photon_hdr)

        # save obs and wave ranges to second extension
        bandpas0, bandpas1 = _np.vstack(self.obs_bandpasses).T
        start, stop = _np.vstack(self.obs_times).T
        obs_nos = range(len(self.obs_bandpasses))
        arys = [obs_nos, bandpas0, bandpas1, start, stop]
        names = [_name_dict['n'], 'bandpas0', 'bandpas1', 'start', 'stop']
        units = [''] + [str(self['w'].unit)]*2 + [str(self['t'].unit)]*2
        formats = ['I'] + ['D']*4
        info_cols = [_fits.Column(array=a, name=n, unit=u, format=fmt)
                     for a,n,u,fmt in zip(arys, names, units, formats)]
        info_hdr = _fits.Header()
        info_hdr['comment'] = 'Bandpass and time coverage of each observation.'
        info_hdu = _fits.BinTableHDU.from_columns(info_cols, header=info_hdr)

        # save obs info as additional headers
        obs_hdus = []
        for item in self.obs_metadata:
            if isinstance(item, _fits.Header):
                hdr = item
            elif hasattr(item, 'iteritems'):
                hdr = _fits.Header(item.iteritems())
            else:
                raise ValueError('FITS file cannot be constructed because Photons object has an improper list of '
                                 'observation metadata. The metadata items must either be pyFITS header objects or '
                                 'have an "iteritems()" method (i.e. be dictionary-like).')
            hdu = _fits.BinTableHDU(header=item)
            obs_hdus.append(hdu)

        hdulist = _fits.HDUList([primary_hdu, photon_hdu, info_hdu] + obs_hdus)
        hdulist.writeto(path, clobber=overwrite)


    @classmethod
    def loadFITS(cls, path):

        # create an empty Photons object
        obj = cls()

        # open file
        hdulist = _fits.open(cls)

        # parse photon data
        photon_hdu = hdulist[1]
        photon_hdr, photons = photon_hdu.header, photon_hdu.data
        obj.time_datum = _time.Time(photon_hdr['zerotime'], format='jd')
        tbl_cols = []
        for i, key in enumerate(photons.names):
            unit = photon_hdr['TUNIT{}'.format(i)]
            unit = _u.Unit(unit)
            name = cls._alternate_names[key]
            col = _tbl.Column(data=photons[key], name=name, unit=unit)
            tbl_cols.append(col)
        obj.photons = _tbl.Table(tbl_cols)

        # parse observation time and wavelength ranges
        info = hdulist[2].data
        obs_nos = info[_name_dict['n']]
        def parse_info(col0, col1):
            ary = _np.array([info[col0], info[col1]]).T
            return [ary[obs_nos == i, :] for i in range(obs_nos.max())]
        obj.obs_bandpasses = parse_info('bandpas0', 'bandpas1')
        obj.obs_times = parse_info('start', 'stop')

        # parse observation metadata
        obj.obs_metadata = [hdu.header for hdu in hdulist[3:]]

        return obj






    # ---------
    # DATA MANIPULATION METHODS
    def set_time_datum(self, new_datum=None):
        """
        Modifies the Photons object in-place to have a new time datum. Default is to set to the time of the earliest
        photon.

        Parameters
        ----------
        new_datum : any object recognized by astropy.time.Time()

        Returns
        -------
        None
        """
        if new_datum is None:
            dt = _time.TimeDelta(self['t'].min(), format=self['t'].unit.to_string())
            new_datum = self.time_datum + dt
        else:
            dt = new_datum - self.time_datum

        dt = dt.to(self['t'].unit).value
        self['t'] -= dt
        self.time_datum = new_datum


    def match_units(self, other):
        """
        Converts the units of each column in self to the units of hte corresponding column in other, if mathcing
        columns are present.

        Parameters
        ----------
        other : Photons object

        Returns
        -------

        """
        for key in self.colnames:
            if key in other.colnames:
                if other[key].unit:
                    unit = other[key].unit
                    self[key].convert_unit_to(unit)


    def add_observations_column(self):
        """
        Adds a column for observation identifiers (to match the index of the observation metadata list item) to self,
        if such a column is not already present.
        """
        if 'n' not in self:
            if len(self.obs_metadata) > 1:
                raise ValueError('Photons cannot be assigned to multiple observations because who the F knows which '
                                 'obseration they belong to?')
            n_ary = _np.zeros(len(self.photons))
            n_col = _tbl.Column(data=n_ary, dtype='i2', name='n')
            self.photons.add_column(n_col)


    def divvy(self, ysignal, yback=[]):
        """
        Provides a simple means of divyying photons into signal and background regions (and adding/updating the
        associated 'r' column, by specifying limits of these regions in the y coordinate.

        Users can implement more complicated divvying schemes (such as changing signal and background region sizes) by
        simply creating their own 'r' column explicitly.

        Parameters
        ----------
        ysignal : 1D or 2D array-like
            [[y00, y01], [y10, y11], ...] giving limits of signal regions
        yback : 1D or 2D array-like
            akin to ysignal, but for background regions

        Returns
        -------

        """

        # groom the input
        ysignal, yback = [_np.reshape(a, [-1, 2]) for a in [ysignal, yback]]

        # join the edges into one list
        edges, isignal, iback, area_ratio = self._get_ribbon_edges(ysignal, yback)

        # determine which band counts are in
        ii = _np.searchsorted(edges, self['y'])

        # add/modify weights in 'r' column
        # TRADE: sacrifice memory with a float weights column versus storing the area ratio and just using integer
        # flags because this allows better flexibility when combining photons from multiple observations
        self['r'] = _np.zeros_like(self['e']) if 'e' in self else _np.zeros(len(self), 'f4')
        signal = reduce(_np.logical_or, [ii == i for i in isignal])
        self['r'][signal] = 1.0
        if len(yback) > 0:
            bkgnd = reduce(_np.logical_or, [ii == i for i in iback])
            self['r'][bkgnd] = -area_ratio


    def squish(self, keep='both'):
        pass






    # --------
    # ANALYSIS METHODS
    def image(self, wbins, ybins):
        # decided not to attempt fluxing -- for HST data this will be problematic if effective areas for the event
        # locations aren't very carefully computed. In that case, the x2ds are the way to go. Maybe this will make
        # sense for other applications in the future
        weights = self['e'] if 'e' in self else None

        counts, wbins, ybins = _np.histogram2d(self['w'], self['y'], bins=[wbins, ybins], weights=weights)
        bintime = self.time_per_bin(wbins)
        rates = counts/bintime[:, _np.newaxis]
        return wbins, ybins, rates


    def spectrum(self, bins, waverange=None, fluxed=False, energy_units='erg', order='all'):
        if order == 'all':
            filter = None
        else:
            filter = (self['o'] == order)

        bin_edges = self._groom_wbins(bins, waverange)
        counts, errors = self._histogram('w', bin_edges, waverange, fluxed, energy_units, filter)

        # divide by bin widths and exposure time to get rates
        bin_exptime = self.time_per_bin(bin_edges)
        bin_exptime[bin_exptime == 0] = _np.nan # use nans to avoid division errors
        bin_widths = _np.diff(bin_edges)
        rates = counts/bin_exptime/bin_widths
        errors = errors/bin_exptime/bin_widths

        # get bin midpoints
        bin_midpts = (bin_edges[:-1] + bin_edges[1:])/2.0

        return bin_edges, bin_midpts, rates, errors


    def spectrum_smooth(self, n, waverange=None, fluxed=False, energy_units='erg'):
        # TODO: check with G230L spectrum from ak sco and see what is going on

        # sort photons and weights in order of wavelength
        weights = self._full_weights(fluxed, energy_units)
        isort = _np.argsort(self['w'])
        w, weights = self['w'][isort], weights[isort]
        if waverange is None:
            waverange = w[[0, -1]]

        # smooth using same process as for lightcurve_smooth
        bin_start, bin_stop, bin_midpts, rates, errors = _smooth_boilerplate(w, weights, n, waverange)

        # divide by time to get rates
        bin_exptimes = self.time_per_bin([bin_start, bin_stop])
        rates = rates/bin_exptimes
        errors = errors/bin_exptimes

        return bin_start, bin_stop, bin_midpts, rates, errors


    def lightcurve(self, time_step, bandpasses, time_range=None, bin_method='elastic', fluxed=False,
                   energy_units='erg'):

        if time_range is None:
            obs_times = _np.vstack(self.obs_times)
            time_range = [obs_times.min(), obs_times.max()]

        # construct time bins. this is really where this method is doing a lot of work for the user in dealing with
        # the exposures and associated gaps
        edges, valid_bins = self._construct_time_bins(time_step, bin_method, time_range)

        inbands = self._bandpass_filter(bandpasses)

        # histogram the counts
        counts, errors = self._histogram('t', edges, time_range, fluxed, energy_units, filter=inbands)

        # get length of each time bin and the bin start and stop
        dt = _np.diff(edges)
        bin_start, bin_stop = edges[:-1], edges[1:]

        # get rid of the bins in between exposures
        counts, errors, dt = [a[valid_bins] for a in [counts, errors, dt, bin_start, bin_stop]]

        # divide by exposure time to get rates
        rates, errors = counts/dt, errors/dt

        # bin midpoints
        bin_midpts = (bin_start + bin_stop)/2.0

        return bin_start, bin_stop, bin_midpts, rates, errors


    def lightcurves_smooth(self, n, bandpasses, time_range=None, fluxed=False, energy_units='erg'):
        inbands = self._bandpass_filter(bandpasses)

        if time_range:
            in_time_range = (self['t'] >= time_range[0]) & (self['t'] <= time_range[1])
            keep = inbands & in_time_range
        else:
            keep = inbands

        weights = self._full_weights(fluxed, energy_units)

        obs, t, weights = self['n'][keep], self['t'][keep], weights[keep]

        curves =[] # each curve in list will have bin_start, bin_stop, bin_midpts, rates, error
        for i in range(len(self.obs_metadata)):
            from_obs_i = (obs == i)
            if _np.sum(from_obs_i) < n:
                continue
            curves.append(_smooth_boilerplate(t[from_obs_i], weights[from_obs_i], n, time_range))

        # sneaky code to take the list of curves and combine them
        bin_start, bin_stop, bin_midpt, rates, error = [_np.hstack(a) for a in zip(curves)]

        return bin_start, bin_stop, bin_midpt, rates, error


    def spectrum_frames(self, bins, time_step, waverange, time_range, bin_method='full', fluxed=False,
                        energy_units='erg'):

        # groom wavelength bins
        bin_edges = self._groom_wbins(bins, waverange)
        bin_midpts = (bin_edges[1:] + bin_edges[:-1])/2.0

        # check that wavelength bins are fully within ranges of observations
        within_obs = self.check_wavelength_coverage([bin_edges[[0,-1]]])[0]
        if not within_obs:
            raise ValueError('Bins must fall within the wavelength range of all observations.')

        # get start and stop of all the time steps
        time_edges, valid_time_bins = self._construct_time_bins(time_step, bin_method, time_range)
        starts, stops = time_edges[:-1][valid_time_bins], time_edges[1:][valid_time_bins]
        time_midpts = (starts + stops)/2.0

        spectra = []
        for start, stop in zip(starts, stops):
            in_time_range = (self['t'] >= start) & (self['t'] < stop)
            density, errors = self._histogram('w', bin_edges, waverange, fluxed, energy_units, filter=in_time_range)
            dt = stop - start
            rates, errors = density/dt, errors/dt
            spectra.append([rates, errors])
        rates, errors = map(_np.array, zip(spectra))

        return starts, stops, time_midpts, bin_edges, bin_midpts, rates, errors





    # ---------
    # UTILITIES

    def check_wavelength_coverage(self, bandpasses):
        covered = []
        for band in bandpasses:
            waveranges = _np.vstack(self.obs_bandpasses)
            beyond_range = [band[0] < waverange[0] or band[1] > waverange[1] for waverange in waveranges]
            if any(beyond_range):
                covered.append(False)
            else:
                covered.append(True)
        return _np.array(covered)


    def total_time(self):
        """
        Compute the cumulative time of the observations.

        Returns
        -------
        T : float
        """
        obs_times = _np.vstack(self.obs_times)
        return _np.sum(obs_times[:, 1] - obs_times[:, 0])


    def time_per_bin(self, bin_edges):
        bin_edges = _np.asarray(bin_edges)

        # parse left and right bin edges
        if bin_edges.ndim == 1:
            w0, w1 = bin_edges[:-1], bin_edges[1:]
        if bin_edges.ndim == 2:
            if bin_edges.shape[1] == 2:
                w0, w1 = bin_edges.T
            else:
                w0, w1 = bin_edges
        widths_full = w1 - w0 # full bin wdiths

        t = _np.zeros(len(bin_edges) - 1, 'f8')
        time_ranges, wave_ranges = map(_np.vstack,[self.obs_times, self.obs_bandpasses])
        for tr, wr in zip(time_ranges, wave_ranges):
            # total exposure time for observation
            dt = tr[1] - tr[0]

            # shift left edges of bins left of wr to wr[0], same for right edges right of wr[1]
            w0[w0 < wr[0]] = wr[0]
            w1[w1 > wr[1]] = wr[1]

            # recompute bin widths, now those fully outside of wr will have negative value, so set them to 0.0
            widths_partial = w1 - w0
            widths_partial[widths_partial < 0] = 0.0

            # compute and add exposure times, using fractional of bin width after adjusting to wr vs original bin
            # widths. this will cause bins outside of wr to get 0 exposure time, those inside to get full exposure
            # time, and partial bins to get partial exposure time
            fractions = widths_partial / widths_full
            t += fractions*dt

        return t





    # --------
    # HIDDEN METHODS
    def _get_proper_key(self, key):
        key = key.lower()
        if key in self._alternate_names.values():
            return key
        elif key in self._alternate_names:
            return self._alternate_names[key]
        else:
            raise KeyError('{} not recognized as a field name'.format(key))


    def _get_ribbon_edges(self, ysignal, yback):
        # join ranges into one list
        ys = list(ysignal) + list(yback)

        #check for bad input
        ys = sorted(ys, key=lambda a: a[0])
        ys = _np.array(ys)
        if any(ys[:-1, 1] > ys[1:, 0]):
            raise ValueError('There is overlap in the signal and background regions. That\'s a no-no.')

        # discard duplicate values (i.e. where a signal and background region share an edge)
        edges = _np.unique(ys)

        ymax = self['y'].max()
        ymin = self['y'].min()
        if ymax < edges.max() or ymin > edges.min():
            raise ValueError('Extraction ribbons include areas beyond the range of the counts.')
    
        # find where signal band is in sorted edges
        ysignal_mids = (ysignal[:,0] + ysignal[:,1])/2.0
        isignal = _np.searchsorted(edges, ysignal_mids)
    
        if yback is not None:
            # find area ratio of signal to background
            area_signal = _np.sum(_np.diff(ysignal, axis=1))
            area_back = _np.sum(_np.diff(yback, axis=1))
            area_ratio = float(area_signal)/area_back
    
            # find where background bands are in the sorted edges
            yback_mids = (yback[:,0] + yback[:,1]) / 2.0
            iback = _np.searchsorted(edges, yback_mids)
        else:
            iback, area_ratio = None, None
    
        return edges, isignal, iback, area_ratio


    def _compute_epera(self, units='erg'):
        """
        Computes energy per effective area, applying weights (if available). No distinction is made between
        background and signal counts.

        Returns
        -------

        """
        if 'a' not in self:
            raise ValueError('Photons must have effective area data to permit the computation of fluxes.')

        energy = _const.h * _const.c / self['w']
        energy = energy.to(units)
        epera = energy / self['a']
        return epera


    def _full_weights(self, fluxed=False, energy_units='erg'):
        if not ('e' in self or 'r' in self) and not fluxed:
            return None
        else:
            weights = _np.ones(len(self), 'f8')
            if 'e' in self: weights *= self['e']
            if 'r' in self: weights *= self['r']
            if fluxed:
                weights *= self._compute_epera(units=energy_units)
            return weights


    def _groom_wbins(self, wbins, wrange=None):
        if wrange is None:
            wranges = _np.vstack(self.obs_bandpasses)
            wrange = [wranges.min(), wranges.max()]
        return _groom_bins(wbins, wrange)


    def _groom_ybins(self, ybins):
        rng = [self['y'].min(), self['y'].max()]
        return _groom_bins(ybins, rng)


    def _construct_time_bins(self, time_step, bin_method, time_range):
        # check for valid input
        validspecs = ['elastic', 'full', 'partial']
        if bin_method not in validspecs:
            raise ValueError('binspec must be one of {}'.format(validspecs))

        # contruct bins for each exposure according to binspec
        dt = time_step
        edges, valid = [], []
        marker = 0
        for rng in _np.vstack(self.obs_times):
            # adjust range to fit time_range if necessary
            if rng[0] >= time_range[1] or rng[1] <= time_range[0]:
                continue
            if rng[0] < time_range[0]:
                rng[0] = time_range[0]
            if rng[1] > time_range[1]:
                rng[1] = time_range[1]

            # make bins fot the obseration
            span = rng[1] - rng[0]
            mid = (rng[0] + rng[1]) / 2.0
            n_exact = span/dt
            if bin_method == 'elastic':
                n = _np.round(n_exact) if n_exact > 1 else 1
                obs_bins = _np.linspace(rng[0], rng[1], n+1)
            else:
                if bin_method == 'full':
                    n = _np.floor(n_exact)
                if bin_method == 'partial':
                    n = _np.ceil(n_exact)
                if n == 0:
                    continue
                start, stop = mid - n*span/2.0, mid + n*span/2.0
                obs_bins = _np.arange(start, stop+dt, dt)

            # add bins to the list
            edges.extend(obs_bins)
            valid.extend(range(len(obs_bins)-1))
            marker += len(obs_bins)

        return map(_np.array, [edges, valid])


    def _bandpass_filter(self, bandpasses):

        # groom bands input
        bands = _np.array(bandpasses)
        if bands.ndim == 1:
            bands = _np.reshape(bands, [-1, 2])

        # check that all bandpasses are fully within every observation's range
        self.check_wavelength_coverage(bands)

        # put bands in order of wavelength
        order = _np.argsort(bands, 0)[:,0]
        bands = bands[order]
        band_edges = _np.ravel(bands)
        if any(band_edges[1:] < band_edges[:-1]):
            raise ValueError('Wavelength bands cannot overlap.')

        # identify photons with wavelengths that fall in the wavelength bands
        i = _np.searchsorted(band_edges, self['w'])
        inbands = (i % 2 == 1)

        return inbands


    def _histogram(self, dim, bin_edges, rng, fluxed, energy_units, filter=None):

        if filter is None:
            filter = _np.ones(len(self), bool)
        x = self[dim][filter]
        weights = self._full_weights(fluxed, energy_units)
        weights = weights[filter]
        counts = _np.histogram(x, bins=bin_edges, range=rng, weights=weights)[0]
        variances = _np.histogram(x, bins=bin_edges, range=rng, weights=weights**2)[0]

        # make sure zero or negative-count bins have conservative errors
        if _np.any(counts) <= 0:
            signal = self['r'][filter] > 0 if  'r' in self else _np.ones(len(x), bool)
            signal_counts = _np.histogram(x[signal], bins=bin_edges, range=rng)[0]
            signal_counts_weighted = _np.histogram(x[signal], bins=bin_edges, range=rng, weights=weights[signal])[0]
            avg_weight = signal_counts_weighted/signal_counts
            min_variance = avg_weight**2
            replace = (counts <= 0) & (variances < min_variance)
            variances[replace] = min_variance[replace]

        errors = _np.sqrt(variances)

        return counts, errors







def _groom_bins(bins, rng):
    # if bins is a float, parse the full range of observations into bins using that float as the bin width,
    # with bins on each end only partially covered by observations
    if isinstance(bins, float):
        dw = bins
        span = rng[1] - rng[0]
        wmid = (rng[0] + rng[1])/2.0
        n = _np.ceil(span / bins)
        w0 = wmid - n/2.0*dw
        w1 = wmid + n/2.0*dw
        edges = _np.arange(w0, w1+dw, dw)

    # if bins is an integer, divide the full range observations into that many bins
    elif isinstance(bins, int):
        n = bins
        edges = _np.linspace(rng[0], rng[1], n+1)

    # if bins is iterable, just give it back
    elif hasattr(bins, '__iter__'):
        edges = bins
    else:
        raise ValueError('Input bins must be either an integer, float, or iterable.')
    return edges


def _smooth_sum(x, n):
    """
    Compute an n-point moving average of the data in vector x. Result will have a length of len(x) - (n-1). Using
    save avoids the arithmetic overflow and accumulated errors that can result from using numpy.cumsum, though cumsum
    is (probably) faster.
    """
    m = len(x)
    result = _np.zeros(m - (n-1))
    for i in xrange(n):
        result += x[i:(m-n+i+1)]
    return result


def _smooth_bins(x, n, xrange=None):
    if xrange is None:
        xrange = x.min(), x.max()
    temp = _np.insert(x, [0, len(x)], xrange)
    xmids = (temp[:-1] + temp[:1]) / 2.0
    bin_start = xmids[:-n]
    bin_stop = xmids[n:]
    return bin_start, bin_stop


def _smooth_boilerplate(x, weights, n, xrange=None):
    counts = _smooth_sum(weights, n)
    errors = _np.sqrt(_smooth_sum(weights**2, n))
    bin_start, bin_stop = _smooth_bins(x, xrange)

    # divide by bin wdiths and exposure times to get rates
    bin_widths = bin_stop - bin_start
    rates = counts/bin_widths
    errors = errors/bin_widths

    # bin midpoints
    bin_midpts = (bin_start + bin_stop)/2.0

    return bin_start, bin_stop, bin_midpts, rates, errors


def _inbins(bins, values):
    bin_edges = _np.ravel(bins)
    bin_no = _np.searchsorted(bin_edges, values)
    return bin_no % 2 == 1


# import some submodules. I put these down here because they themselves import and use __init__. This is probably
# bad. Maybe the photons class should be split out from __init__...
import hst