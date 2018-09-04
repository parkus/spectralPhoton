import copy as _copy

import numpy as _np
import utils
from astropy import time as _time, table as _tbl, units as _u, constants as _const, table as _table
from astropy.io import fits as _fits
from crebin import rebin as _rebin
from matplotlib import pyplot as plt, pyplot as _pl


def _FITSformat(dtype):
    dstr = str(dtype)
    dstr = dstr.replace('>', '')
    if dstr in ['uint8']:
        return 'B'
    if dstr in ['int8', 'i1', 'int16']:
        return 'I'
    if dstr in ['uint16', 'int32', 'i2', 'i4']:
        return 'J'
    if dstr in ['uint32', 'int64', 'i8']:
        return 'K'
    if dstr in ['float32', 'f4']:
        return 'E'
    if dstr in ['float64', 'f8']:
        return 'D'
    raise ValueError('Not sure waht to do with the {} data type.'.format(dstr))


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

    #region OBJECT HANDLING
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

        if 'photons' in kwargs:
            self.photons = kwargs['photons']
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


    _ovr_doc = ('"overlap_handling : "adjust Aeff"|"clip"\n'
                '\t    How to handle multiple observations that overlap in wavelength.\n'
                '\t      - If "clip," photons from the observation with fewer photons in the overlap are removed.\n'
                '\t      - If "adjust Aeff," the effective area estimate at the wavelengths of the affected photons '
                'are increased as appropriate.')
    def merge_like_observations(self, overlap_handling="adjust Aeff", min_rate_ratio=0.5):
        """
        Merge observations that have the same exposure times in place.

        Parameters
        ----------
        {ovr}

        If their bandpass ranges overlap, then the photons
        in the overlap get de-weighted due to the "extra" observing time that isn't otherwise accounted for.

        Returns
        -------
        None (operation done in place)

        """
        def get_signal():
            if 'r' in self:
                signal = self['r'] > 0
            else:
                signal = _np.ones(len(self), bool)
            return signal
        signal = get_signal()

        i = 0
        while i < len(self.obs_metadata)-1:
            j = i + 1
            while j < len(self.obs_metadata) - 1:
                # if observations have same exposure starts and ends
                if _np.all(self.obs_times[i] == self.obs_times[j]):
                    i_photons = self['n'] == i
                    j_photons = self['n'] == j
                    overlap = utils.rangeset_intersect(self.obs_bandpasses[i], self.obs_bandpasses[j])
                    if len(overlap) > 0:
                        in_overlap = utils.inranges(self['w'], overlap)
                        xi = i_photons & in_overlap
                        xj = j_photons & in_overlap
                        Ni = float(_np.sum(xi & signal))
                        Nj = float(_np.sum(xj & signal))
                        worthwhile = (Ni > 0) and (Nj > 0) and (Ni/Nj > min_rate_ratio) and (Nj/Ni > min_rate_ratio)
                        if overlap_handling == "adjust Aeff" and worthwhile:
                            Ai = self._Aeff_interpolator(filter=xi)
                            Aj = self._Aeff_interpolator(filter=xj)
                            wi = self['w'][xi]
                            wj = self['w'][xj]
                            Ai_at_j = Ai(wj)
                            Aj_at_i = Aj(wi)
                            self['a'][xi] += Aj_at_i
                            self['a'][xj] += Ai_at_j
                        elif overlap_handling == "clip" or not worthwhile:
                            if Ni < Nj:
                                ii, = _np.nonzero(xi)
                                self.photons.remove_rows(ii)
                                self.obs_bandpasses[i] = utils.rangeset_subtract(self.obs_bandpasses[i], overlap)
                            else:
                                jj, = _np.nonzero(xj)
                                self.photons.remove_rows(jj)
                                self.obs_bandpasses[j] = utils.rangeset_subtract(self.obs_bandpasses[j], overlap)
                            j_photons = self['n'] == j
                            signal = get_signal()
                        else:
                            raise ValueError("overlap_handling option not recognized.")

                    # associate photons from observation j with i
                    self['n'][j_photons] = i

                    # decrement higher observation numbers (else while loop indexing gets messed up)
                    self['n'][self.photons['n'] > j] -= 1

                    # update properties of observation i
                    self.obs_metadata[i] += self.obs_metadata[j]
                    self.obs_bandpasses[i] = utils.rangeset_union(self.obs_bandpasses[i], self.obs_bandpasses[j])

                    # remove observation j
                    del self.obs_times[j], self.obs_metadata[j], self.obs_bandpasses[j]
                else:
                    j += 1
            i += 1

    merge_like_observations.__doc__ = merge_like_observations.__doc__.format(ovr=_ovr_doc)


    def merge_orders(self, overlap_handling="adjust Aeff"):
        """
        Merge the orders in each observation in place.

        Parameters
        ----------
        {ovr}

        Returns
        -------
        None (operation done in place)

        Notes
        -----
        Merging is accomplished by changing the "region" weights of the photons where there is overlap in wavelength
        to account for the double (or more than double) counting.

        """

        # split into separate photons objects for each observation, and split the orders within that observation into
        # faux separate observations, then merge them
        if len(self.obs_times) > 1:
            separate = [self.get_obs(i) for i in range(len(self.obs_times))]
        else:
            separate = [self]
        for obj in separate:
            temp_meta = obj.obs_metadata
            order_ranges = obj.obs_bandpasses[0]
            Norders = len(order_ranges)
            obj.obs_metadata = [0]*Norders
            obj.obs_times *= Norders
            obj.obs_bandpasses = [order_ranges[[i]] for i in range(len(order_ranges))]
            obj.photons['n'] = obj.photons['o'] - _np.min(obj.photons['o'])
            obj.photons.remove_column('o')
            obj.merge_like_observations(overlap_handling=overlap_handling)
            obj.obs_metadata = temp_meta
            obj.photons.remove_column('n')
            obj.obs_bandpasses = [_np.vstack(obj.obs_bandpasses)]
            obj.obs_times = [obj.obs_times[0]]

        all = sum(separate[1:], separate[0])
        self.obs_times = all.obs_times
        self.obs_bandpasses = all.obs_bandpasses
        self.obs_metadata = all.obs_metadata
        self.photons = all.photons
    merge_orders.__doc__ = merge_orders.__doc__.format(ovr=_ovr_doc)

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
        photons = _tbl.vstack([self.photons, other.photons])

        # leave it to the user to deal with sorting and grouping and dealing with overlap as they see fit :)
        obs_metadata = self.obs_metadata + other.obs_metadata
        obs_times = list(self.obs_times) + list(other.obs_times)
        obs_bandpasses = list(self.obs_bandpasses) + list(other.obs_bandpasses)

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
        """

        Parameters
        ----------
        path
        overwrite

        Returns
        -------

        """

        primary_hdu = _fits.PrimaryHDU()

        # save photontable to first extension
        photon_cols = []
        for colname in self.photons.colnames:
            tbl_col = self[colname]
            name = _name_dict.get(colname, colname)
            format = _FITSformat(tbl_col.dtype)
            fits_col = _fits.Column(name=name, format=format, array=tbl_col.data, unit=str(tbl_col.unit))
            photon_cols.append(fits_col)
        photon_hdr = _fits.Header()
        photon_hdr['zerotime'] = (self.time_datum.jd, 'julian date of time = 0.0')
        photon_hdu = _fits.BinTableHDU.from_columns(photon_cols, header=photon_hdr)

        # save obs metadata, time ranges, bandpasses to additional extensions
        obs_hdus = []
        for meta, times, bands in zip(self.obs_metadata, self.obs_times, self.obs_bandpasses):
            # check that meta can be saved as a fits header
            if isinstance(meta, _fits.Header):
                hdr = meta
            elif hasattr(meta, 'iteritems'):
                hdr = _fits.Header(meta.iteritems())
            else:
                raise ValueError('FITS file cannot be constructed because Photons object has an improper list of '
                                 'observation metadata. The metadata items must either be pyFITS header objects or '
                                 'have an "iteritems()" method (i.e. be dictionary-like).')

            # save obs time and wave ranges to each extension
            bandpas0, bandpas1 = bands.T
            start, stop = times.T
            arys = [bandpas0, bandpas1, start, stop]
            names = ['bandpas0', 'bandpas1', 'start', 'stop']
            units = [str(self['w'].unit)]*2 + [str(self['t'].unit)]*2
            formats = ['{}D'.format(len(a)) for a in arys]
            info_cols = [_fits.Column(array=a.reshape([1,-1]), name=n, unit=u, format=fmt)
                         for a,n,u,fmt in zip(arys, names, units, formats)]

            hdu = _fits.BinTableHDU.from_columns(info_cols, header=meta)
            obs_hdus.append(hdu)

        # save all extensions
        hdulist = _fits.HDUList([primary_hdu, photon_hdu] + obs_hdus)
        hdulist.writeto(path, overwrite=overwrite)


    @classmethod
    def loadFITS(cls, path):
        """

        Parameters
        ----------
        path

        Returns
        -------
        Photons object
        """

        # create an empty Photons object
        obj = Photons()

        # open file
        hdulist = _fits.open(path)

        # parse photon data
        photon_hdu = hdulist[1]
        photon_hdr, photons = photon_hdu.header, photon_hdu.data
        obj.time_datum = _time.Time(photon_hdr['zerotime'], format='jd')
        tbl_cols = []
        for i, key in enumerate(photons.names):
            unit = photon_hdr['TUNIT{}'.format(i+1)]
            if unit not in [None, "None"]:
                unit = _u.Unit(unit)
            name = cls._alternate_names.get(key, key.lower())
            col = _tbl.Column(data=photons[key], name=name, unit=unit)
            tbl_cols.append(col)
        obj.photons = _tbl.Table(tbl_cols)

        # parse observation time and wavelength ranges
        def parse_bands_and_time(extension):
            bandpas0, bandpas1, start, stop = [extension.data[s] for s in ['bandpas0', 'bandpas1', 'start', 'stop']]
            bands = _np.vstack([bandpas0, bandpas1]).T
            times = _np.vstack([start, stop]).T
            bands.reshape(-1,2)
            times = times.reshape(-1,2)
            return bands, times

        # parse observation metadata
        obj.obs_metadata = [hdu.header for hdu in hdulist[2:]]

        # parse times and bands
        pairs = map(parse_bands_and_time, hdulist[2:])
        bands, times = zip(*pairs)
        obj.obs_times = times
        obj.obs_bandpasses = bands

        return obj
    #endregion


    #region DATA MANIPULATION METHODS
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

        # ensure appropriate precision is maintained
        ddt = _np.diff(self['t'])
        dtmin = ddt[ddt > 0].min()
        max_bit = _np.log2(self['t'][-1] + abs(dt))
        min_bit = _np.log2(dtmin)
        need_bits = _np.ceil(max_bit - min_bit) + 3
        need_bytes = _np.ceil(need_bits/8.)
        if need_bytes > 8:
            raise ValueError('Resetting the time atum of this observation by {} {} will result in loss of numerical '
                             'precision of the photon arrival times.'.format(dt, self['t'].unit))
        use_bytes = have_bytes = int(self['t'].dtype.str[-1])
        while need_bytes > use_bytes and use_bytes < 8:
            use_bytes *= 2
        if use_bytes != have_bytes:
            new_dtype = 'f' + str(use_bytes)
            self['t'] = self['t'].astype(new_dtype)

        self['t'] -= dt
        self.time_datum = new_datum
        self.obs_times = [t - dt for t in self.obs_times]


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
        for key in self.photons.colnames:
            if key in other.photons.colnames:
                if other[key].unit:
                    unit = other[key].unit
                    if str(unit).lower() == 'none' and str(self[key].unit).lower() == 'none':
                        continue
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


    def divvy(self, ysignal, yback=(), order=None):
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
        assert order is None or type(order) == int

        # join the edges into one list
        edges, isignal, iback, area_ratio = self._get_ribbon_edges(ysignal, yback)

        # deal just with the photons of appropriate order
        if order is not None:
            filter, = _np.nonzero(self['o'] == order)

        # determine which band counts are in
        y = self['y'] if order is None else self['y'][filter]
        ii = _np.searchsorted(edges, y)

        # add/modify weights in 'r' column
        # TRADE: sacrifice memory with a float weights column versus storing the area ratio and just using integer
        # flags because this allows better flexibility when combining photons from multiple observations
        self['r'] = _np.zeros(len(self), 'f4')
        signal = reduce(_np.logical_or, [ii == i for i in isignal])
        if order is not None: signal = filter[signal]
        self['r'][signal] = 1.0
        if len(yback) > 0:
            bkgnd = reduce(_np.logical_or, [ii == i for i in iback])
            if order is not None: bkgnd = filter[bkgnd]
            self['r'][bkgnd] = -area_ratio


    def squish(self, keep='both'):
        """
        Removes superfluous counts -- those that aren't in a signal region or background region.
        """
        if 'r' not in self:
            raise ValueError('Photon object must have an \'r\' column (specifying a region weight) before it can be '
                             'squished.')
        valid_keeps = ['both'] + ['back', 'background', 'bg'] + ['signal']
        if keep not in valid_keeps:
            raise ValueError('keep parameter must be one of {}'.format(valid_keeps))

        new = self.copy()
        if keep in ['both']:
            superfluous = (self['r'] == 0)
            new.photons = self.photons[~superfluous]
        elif keep in ['back', 'background', 'bg']:
            new.photons = self.photons[self['r'] < 0]
            del new.photons['r']
        elif keep in ['signal']:
            new.photons = self.photons[self['r'] > 0]
            del new.photons['r']
        return new
    #endregion


    # region ANALYSIS METHODS
    def spectrum(self, bins, waveranges=None, fluxed=False, energy_units='erg', order=None, time_ranges=None,
                 bin_method='elastic', background=False):
        """

        Parameters
        ----------
        bins
        waveranges
        fluxed
        energy_units
        order

        Returns
        -------
        bin_edges, bin_midpts, rates, errors

        """
        filter = self._filter_boiler(waveranges=None, time_ranges=time_ranges, order=order)

        bin_edges, i_gaps = self._groom_wbins(bins, waveranges, bin_method=bin_method)
        counts, errors = self._histogram('w', bin_edges, None, fluxed, energy_units, filter, background=background)

        # add nans if there are gaps
        if i_gaps is not None:
            counts[i_gaps] = _np.nan
            errors[i_gaps] = _np.nan

        # divide by bin widths and exposure time to get rates
        bin_exptime = self.time_per_bin(bin_edges, time_ranges=time_ranges)
        bin_exptime[bin_exptime == 0] = _np.nan # use nans to avoid division errors
        bin_widths = _np.diff(bin_edges)
        rates = counts/bin_exptime/bin_widths
        errors = errors/bin_exptime/bin_widths

        # get bin midpoints
        bin_midpts = (bin_edges[:-1] + bin_edges[1:])/2.0

        return bin_edges, bin_midpts, rates, errors


    def spectrum_smooth(self, n, wave_range=None, time_ranges=None, fluxed=False, energy_units='erg', order=None):
        """

        Parameters
        ----------
        n
        wave_range
        time_range
        fluxed
        energy_units

        Returns
        -------
        bin_start, bin_stop, bin_midpts, rates, errors
        """

        # TODO: check with G230L spectrum from ak sco and see what is going on

        filter = self._filter_boiler(waveranges=wave_range, time_ranges=time_ranges, order=order)

        # get pertinent photon info
        weights = self._full_weights(fluxed, energy_units)
        w = self['w']

        # which photons have nonzero weight
        countable = (weights != 0)

        # filter out superfluous photons
        keep = filter & countable
        w, weights = [a[keep] for a in [w, weights]]

        # sort photons and weights in order of wavelength
        isort = _np.argsort(w)
        w, weights = w[isort], weights[isort]
        if wave_range is None:
            wave_range = w[[0, -1]]

        # smooth using same process as for lightcurve_smooth
        bin_start, bin_stop, bin_midpts, rates, errors = _smooth_boilerplate(w, weights, n, wave_range)

        # divide by time to get rates
        bin_exptimes = self.time_per_bin([bin_start, bin_stop], time_ranges)
        rates = rates/bin_exptimes
        errors = errors/bin_exptimes

        return bin_start, bin_stop, bin_midpts, rates, errors


    def lightcurve(self, time_step, bandpasses, time_range=None, bin_method='elastic', fluxed=False,
                   energy_units='erg', nan_between=False, background=False, order=None):
        """

        Parameters
        ----------
        time_step
        bandpasses
        time_range
        bin_method:
            elastic, full, or partial
        fluxed
        energy_units

        Returns
        -------
        bin_start, bin_stop, bin_midpts, rates, errors
        """

        # check that at least some observations cover the input
        covered = self.check_wavelength_coverage(bandpasses)
        if not _np.any(covered):
            raise ValueError('None of the observations cover the provided bandpasses.')

        # construct time bins. this is really where this method is doing a lot of work for the user in dealing with
        # the exposures and associated gaps
        if hasattr(time_step, '__iter__'):
            edges = _np.unique(_np.ravel(time_step))
            mids = utils.midpts(edges)
            valid_times = utils.rangeset_intersect(_np.vstack(self.obs_times), _np.array(time_step))
            valid_bins = utils.inranges(mids, valid_times)
        else:
            edges, valid_bins = self._construct_time_bins(time_step, bin_method, time_range, bandpasses)

        inbands = self._bandpass_filter(bandpasses, check_coverage=False)
        filter = self._filter_boiler(waveranges=None, time_ranges=None, order=order)
        filter = filter & inbands

        # histogram the counts
        counts, errors = self._histogram('t', edges, time_range, fluxed, energy_units, filter=filter,
                                         background=background)

        # get length of each time bin and the bin start and stop
        dt = _np.diff(edges)
        bin_start, bin_stop = edges[:-1], edges[1:]

        # get rid of the bins in between exposures
        if nan_between:
            betweens = _np.ones_like(counts, bool)
            betweens[valid_bins] = False
            for a in [counts, errors, dt, bin_start, bin_stop]:
                a[betweens] = _np.nan
        else:
            counts, errors, dt, bin_start, bin_stop = [a[valid_bins] for a in [counts, errors, dt, bin_start, bin_stop]]

        if counts.size == 0:
            return [_np.array([]) for _ in range(5)]


        # divide by exposure time to get rates
        rates, errors = counts/dt, errors/dt

        # bin midpoints
        bin_midpts = (bin_start + bin_stop)/2.0

        return bin_start, bin_stop, bin_midpts, rates, errors


    def lightcurve_smooth(self, n, bandpasses, time_range=None, fluxed=False, energy_units='erg', nan_between=False,
                          independent=False, order=None):
        """

        Parameters
        ----------
        n
        bandpasses
        time_range
        fluxed
        energy_units
        nan_between
            Add NaN points between each observation. Useful; for plotting because it breaks the plotted line between
            exposures.
        independent :
            Return only ever nth point in each exposure such that the points are statistically independent.

        Returns
        -------
        bin_start, bin_stop, bin_midpt, rates, error
        """
        # FIXME there is a bug where fluxing doesn't work when time range is set

        # check that at least some observations cover the input
        covered = self.check_wavelength_coverage(bandpasses)
        if not _np.any(covered):
            raise ValueError('None of the observations cover the provided bandpasses.')

        # get pertinent photon info
        weights = self._full_weights(fluxed, energy_units)
        t = self['t']
        obs = self['n'] if 'n' in self else _np.zeros(len(t), bool)

        # which photons are in wavelength bandpasses
        inbands = self._bandpass_filter(bandpasses, check_coverage=False)

        # which photons are in specified time range and order
        filter = self._filter_boiler(waveranges=None, time_ranges=time_range, order=order)

        # which photons have nonzero weight
        countable = (weights != 0)

        # filter superfluous photons
        keep = inbands & filter & countable
        t, obs, weights = [a[keep] for a in [t, obs, weights]]
        isort = _np.argsort(t)
        t, obs, weights = [a[isort] for a in [t, obs, weights]]

        curves =[] # each curve in list will have bin_start, bin_stop, bin_midpts, rates, error
        for i in range(len(self.obs_metadata)):
            from_obs_i = (obs == i)
            _t, _weights = t[from_obs_i], weights[from_obs_i]
            for rng in self.obs_times[i]:
                inrng = (_t > rng[0]) & (_t < rng[1])
                if _np.sum(inrng) < n:
                    continue
                curve = _smooth_boilerplate(_t[inrng], _weights[inrng], n, time_range, independent)
                if nan_between:
                    curve = [_np.append(a, _np.nan) for a in curve]
                curves.append(curve)

        # sneaky code to take the list of curves and combine them
        if len(curves) > 0:
            bin_start, bin_stop, bin_midpt, rates, error = [_np.hstack(a) for a in zip(*curves)]
        else:
            bin_start, bin_stop, bin_midpt, rates, error = [_np.array([]) for _ in range(5)]

        return bin_start, bin_stop, bin_midpt, rates, error


    def spectrum_frames(self, bins, time_step, waveranges=None, time_range=None, w_bin_method='full',
                        t_bin_method='full', fluxed=False, energy_units='erg', order=None):
        """

        Parameters
        ----------
        bins
        time_step
        waverange
        time_range
        bin_method
        fluxed
        energy_units

        Returns
        -------
        starts, stops, time_midpts, bin_edges, bin_midpts, rates, errors
        """
        kws = dict(fluxed=fluxed, energy_units=energy_units, order=order)

        # make wbins ahead of time so this doesn't happen in loop
        bin_edges, i_gaps = self._groom_wbins(bins, waveranges, bin_method=w_bin_method)
        kws['bins'] = bin_edges

        # get start and stop of all the time steps
        if hasattr(time_step, '__iter__'):
            print 'I see that you gave user supplied time bins. That is fine, but note that no checks will be ' \
                  'performed to ensure wavlength and time coverage in that range. And any time_range parameter will ' \
                  'be ignored.'
            starts, stops = _np.asarray(time_step).T
        else:
            time_edges, valid_time_bins = self._construct_time_bins(time_step, t_bin_method, time_range)
            starts, stops = time_edges[:-1][valid_time_bins], time_edges[1:][valid_time_bins]
        time_midpts = (starts + stops)/2.0

        spectra = []
        for time_range in zip(starts, stops):
            kws['time_ranges'] = time_range
            spectra.append(self.spectrum(**kws))
        bin_edges, bin_midpts, rates, errors = map(_np.array, zip(*spectra))
        rates[:,i_gaps] = _np.nan
        errors[:,i_gaps] = _np.nan

        return starts, stops, time_midpts, bin_edges, bin_midpts, rates, errors

    def continuum_subtracted_lightcurves(self, dt, dw, continuum_bands, lc_bands, poly_order, fluxed=False,
                                         energy_units='erg', time_range=None, w_bin_method='elastic',
                                         t_bin_method='elastic'):
        kws = dict(fluxed=fluxed, energy_units=energy_units)

        t0, t1, t, we, w, f, e = self.spectrum_frames(dw, dt, waveranges=continuum_bands, w_bin_method=w_bin_method,
                                                      t_bin_method=t_bin_method, time_range=time_range, **kws)
        tbins = _np.array([t0, t1]).T
        good = ~_np.isnan(f[0])
        wbins = _np.array([we[0,:-1][good], we[0,1:][good]]).T
        polyfuncs = []
        for i in range(len(t)):
            _, _, pf = utils.polyfit_binned(wbins, f[i,good], e[i,good], poly_order)
            polyfuncs.append(pf)

        # tf, cf, lf for total flux, continuum flux, line flux
        lfs, les = [], []
        for band in lc_bands:
            _, _, _, tf, te = self.lightcurve(tbins, band, bin_method=t_bin_method, **kws)
            lf, le = [_np.zeros_like(tf) for _ in range(2)]
            for i in range(len(t)):
                cf, cfe = polyfuncs[i](band)
                cf = _np.sum(cf)
                cfe = _np.sqrt(_np.sum(cfe**2))
                lf[i] = tf[i] - cf
                le[i] = _np.sqrt(te[i]**2 + cfe**2)
            lfs.append(lf)
            les.append(le)

        return t0, t1, t, lfs, les



    def average_rate(self, bands, timeranges=None, fluxed=False, energy_units='erg', order=None):
        if timeranges is None:
            timeranges = _np.vstack(self.obs_times)
        t0, t1, t, f, e = self.lightcurve(timeranges, bands, fluxed=fluxed, energy_units=energy_units, order=order)
        dt = t1 - t0
        return _np.sum(dt * f) / _np.sum(dt)

    #endregion


    #region UTILITIES
    def get_obs(self, obsno):
        if 'n' not in self:
            self.photons['n'] = 0
        new = Photons()
        new.photons = self.photons[self.photons['n'] == obsno]
        new.photons['n'] = 0
        new.time_datum = self.time_datum
        new.obs_bandpasses = [self.obs_bandpasses[obsno]]
        new.obs_times = [self.obs_times[obsno]]
        new.obs_metadata = [self.obs_metadata[obsno]]
        return new


    def which_obs(self, t):
        obsno = _np.ones(t.shape, 'i2')*(-1)
        for n, tranges in enumerate(self.obs_times):
            i = _np.searchsorted(tranges.ravel(), t)
            inobs = (i % 2) == 1

            if not _np.all(obsno[inobs] == -1):
                raise ValueError('Some observation time ranges overlap at the input times, so these times cannot be '
                                 'uniquely associated with an observation.')

            obsno[inobs] = n

        return obsno


    def image(self, xax, yax, bins, weighted=False, scalefunc=None, show=True):
        """

        Parameters
        ----------
        wbins
        ybins

        Returns
        -------
        wbins, ybins, rates
        """
        # decided not to attempt fluxing -- for HST data this will be problematic if effective areas for the event
        # locations aren't very carefully computed. In that case, the x2ds are the way to go. Maybe this will make
        # sense for other applications in the future
        weights = self['e'] if ('e' in self and weighted) else None
        counts, xbins, ybins = _np.histogram2d(self[xax], self[yax], bins=bins, weights=weights)

        if type(scalefunc) in [int, float]:
            pow = scalefunc
            scalefunc = lambda x: x**pow
        if scalefunc is None:
            scaled_counts = counts
        else:
            scaled_counts = scalefunc(counts)

        if show:
            extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
            plt.imshow(scaled_counts[:,::-1].T, extent=extent, aspect='auto')

        return counts, xbins, ybins


    def check_wavelength_coverage(self, bandpasses):
        """
        Returns a bollean array where the first dimension corresponds to the bnadpass and the second to the
        observation where observations that fully include the bandpass are marked True.

        Parameters
        ----------
        bandpasses: 2D array like

        Returns
        -------
        covered: 2D boolean array like

        """

        covered_bands = []
        for band in bandpasses:
            covering_obs = []
            for obs_bands in self.obs_bandpasses:
                covering = _np.any((band[0] >= obs_bands[:,0]) & (band[1] <= obs_bands[:,1]))
                covering_obs.append(covering)
            covered_bands.append(covering_obs)

        return _np.array(covered_bands, bool)


    def total_time(self):
        """
        Compute the cumulative time of the observations.

        Returns
        -------
        T : float
        """
        obs_times = self.clean_obs_times()
        return _np.sum(obs_times[:, 1] - obs_times[:, 0])


    def time_per_bin(self, bin_edges, time_ranges=None):
        """

        Parameters
        ----------
        bin_edges

        Returns
        -------
        t
        """
        bin_edges = _np.asarray(bin_edges)
        if time_ranges is not None:
            time_ranges = _np.asarray(time_ranges)
            if time_ranges.ndim != 2:
                time_ranges = _np.reshape(time_ranges, [-1, 2])
            time_ranges = time_ranges[_np.argsort(time_ranges[:,0]), :]

        # parse left and right bin edges
        if bin_edges.ndim == 1:
            w0, w1 = bin_edges[:-1], bin_edges[1:]
        if bin_edges.ndim == 2:
            if bin_edges.shape[1] == 2:
                w0, w1 = bin_edges.T
            else:
                w0, w1 = bin_edges
        widths_full = w1 - w0 # full bin wdiths

        t = _np.zeros(len(w0), 'f8')
        for tranges, wranges  in zip(self.obs_times, self.obs_bandpasses):
            tranges = _np.copy(tranges)
            if time_ranges is not None:
                tranges = utils.rangeset_intersect(time_ranges, tranges, presorted=True)
            if len(tranges) == 0:
                continue

            # total exposure time for observation
            dt = _np.sum(tranges[:,1] - tranges[:,0])

            # avoid double-counting of overlapping orders. (user must call merge_orders t use multiple orders)
            if len(wranges) > 1:
                isort = _np.argsort(wranges[:,0])
                wranges = wranges[isort,:]
                wranges = reduce(utils.rangeset_union, wranges[1:], wranges[:1])

            for wr in wranges:
                # shift left edges of bins left of wr to wr[0], same for right edges right of wr[1]
                # use copies to avoid modfying input bin_edges
                _w0, _w1 = w0.copy(), w1.copy()
                _w0[w0 < wr[0]] = wr[0]
                _w1[w1 > wr[1]] = wr[1]

                # recompute bin widths, now those fully outside of wr will have negative value, so set them to 0.0
                widths_partial = _w1 - _w0
                widths_partial[widths_partial < 0] = 0.0

                # compute and add exposure times, using fractional of bin width after adjusting to wr vs original bin
                # widths. this will cause bins outside of wr to get 0 exposure time, those inside to get full exposure
                # time, and partial bins to get partial exposure time
                fractions = widths_partial / widths_full
                t += fractions*dt

        return t


    def clean_obs_times(self, bandpasses=None):
        # stack and sort all of the time ranges
        if bandpasses is None:
            obs_times_lst = self.obs_times
        else:
            covered = self.check_wavelength_coverage(bandpasses)
            covered = _np.all(covered, 0)
            if not _np.any(covered):
                raise ValueError('No observations cover the provided bandpasses.')
            obs_times_lst = [self.obs_times[i] for i in _np.nonzero(covered)[0]]

        obs_times = _np.vstack(obs_times_lst)
        isort = _np.argsort(obs_times[:,0])
        obs_times = obs_times[isort, :]

        # loop through dealing with overlap when it occurs
        clean_times = [obs_times[0]]
        for rng in obs_times[1:]:
            last = clean_times[-1][-1]

            # if rng overlaps with the end of the last range
            if rng[0] < last:
                # extend the the last range if rng extends beyond its end, otherwise rng is completely overlapped and
                #  can be discarded
                if rng[-1] > last:
                    clean_times[-1][-1] = rng[-1]
            else:
                clean_times.append(rng)

        return _np.vstack(clean_times)


    def abstime(self, t):
        if hasattr(t, 'unit'):
            t = t.to(_u.s)
        else:
            t = t * self['t'].unit
            t = t.to(_u.s)
        t = _time.TimeDelta(t.value, format='sec')
        return self.time_datum + t
    #endregion


    #region HIDDEN METHODS
    def _get_proper_key(self, key):
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        if key in self.photons.colnames:
            return key

        key = key.lower()
        if key in self._alternate_names.values():
            return key
        elif key in self._alternate_names:
            return self._alternate_names[key]
        else:
            raise KeyError('{} not recognized as a field name'.format(key))


    def _get_ribbon_edges(self, ysignal, yback):
        """

        Parameters
        ----------
        ysignal
        yback

        Returns
        -------
        edges, isignal, iback, area_ratio
        """
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
        epera
        """
        if 'a' not in self:
            raise ValueError('Photons must have effective area data to permit the computation of fluxes.')

        energy = _const.h * _const.c / self['w']
        energy = energy.to(units).value
        epera = energy / self['a']
        return epera


    def _full_weights(self, fluxed=False, energy_units='erg'):
        """

        Parameters
        ----------
        fluxed
        energy_units

        Returns
        -------
        weights
        """
        weights = _np.ones(len(self), 'f8')
        if 'e' in self:
            weights *= self['e']
        if 'r' in self:
            weights *= self['r']
        if fluxed:
            weights *= self._compute_epera(units=energy_units)
        return weights


    def _groom_wbins(self, wbins, wranges=None, bin_method='elastic'):
        """

        Parameters
        ----------
        wbins
        wranges

        Returns
        -------
        wbins
        """
        if wranges is None:
            wranges = _np.vstack(self.obs_bandpasses)
            wranges = [wranges.min(), wranges.max()]
        if hasattr(wranges[0], '__iter__'):
            bin_groups, igaps = [], [0]
            for wrange in wranges:
                _wbins = _groom_bins(wbins, wrange, bin_method)
                igaps.append(igaps[-1] + len(_wbins))
                bin_groups.append(_wbins)
            igaps = _np.array(igaps[1:-1]) - 1
            return _np.hstack(bin_groups), igaps
        return _groom_bins(wbins, wranges, bin_method=bin_method), None


    def _groom_ybins(self, ybins):
        rng = [self['y'].min(), self['y'].max()]
        return _groom_bins(ybins, rng, bin_method='partial')


    def _construct_time_bins(self, time_step, bin_method, time_range=None, bandpasses=None):

        if time_range is None:
            obs_times = _np.vstack(self.obs_times)
            time_range = [obs_times.min(), obs_times.max()]

        # check for valid input
        validspecs = ['elastic', 'full', 'partial']
        if bin_method not in validspecs:
            raise ValueError('binspec must be one of {}'.format(validspecs))

        # contruct bins for each exposure according to binspec
        dt = time_step
        edges, valid = [], []
        marker = 0

        for rng in self.clean_obs_times(bandpasses):
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
                start, stop = mid - n*dt/2.0, mid + n*dt/2.0
                obs_bins = _np.arange(start, stop+dt/2.0, dt)
                if bin_method == 'partial':
                    obs_bins[[0,-1]] = rng

            # add bins to the list
            edges.extend(obs_bins)
            valid.extend(range(marker, marker+len(obs_bins)-1))
            marker += len(obs_bins)

        edges = _np.array(edges)
        valid = _np.array(valid, long)
        return edges, valid


    def _bandpass_filter(self, bandpasses, check_coverage=True):
        """

        Parameters
        ----------
        bandpasses

        Returns
        -------
        inbands
        """

        # groom bands input
        bands = _np.array(bandpasses)
        if bands.ndim == 1:
            bands = _np.reshape(bands, [-1, 2])

        # check that all bandpasses are fully within every observation's range
        if check_coverage and not _np.all(self.check_wavelength_coverage(bands)):
            raise ValueError('Some bandpasses fall outside of the observation ranges.')

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


    def _histogram(self, dim, bin_edges, rng, fluxed, energy_units, filter=None, background=False):
        """

        Parameters
        ----------
        dim
        bin_edges
        rng
        fluxed
        energy_units
        filter

        Returns
        -------
        counts, errors
        """

        x = self[dim]
        if background:
            if 'r' not in self:
                raise ValueError('Background region not defined for this photon set.')
            if fluxed:
                raise ValueError('Does not make sense to compute a fluxed background lightcurve.')
            weights = self['r'] < 0
            weights = weights.astype('f4')
        else:
            weights = self._full_weights(fluxed, energy_units)

        if rng is None:
            rng = bin_edges[[0,-1]]
        inrng = (x > rng[0]) & (x < rng[1])

        if filter is None:
            filter = _np.ones(len(self), bool)

        keep = filter & (weights != 0) & inrng

        x = x[keep]
        weights = weights[keep]
        weights[_np.isnan(weights)] = 0.

        counts = _np.histogram(x, bins=bin_edges, range=rng, weights=weights)[0]
        variances = _np.histogram(x, bins=bin_edges, range=rng, weights=weights**2)[0]

        # make sure zero or negative-count bins have conservative errors
        if _np.any(counts <= 0):
            if background:
                variances[variances == 0] = 1
            else:
                signal = self['r'][keep] > 0 if 'r' in self else _np.ones(len(x), bool)
                signal_counts = _np.histogram(x[signal], bins=bin_edges, range=rng)[0]
                signal_counts_weighted = _np.histogram(x[signal], bins=bin_edges, range=rng, weights=weights[signal])[0]
                zeros = (signal_counts == 0)
                if any(zeros):
                    signal_counts[zeros] = 1.0
                    bin_midpts = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    signal_counts_weighted[zeros] = _np.interp(bin_midpts[zeros], bin_midpts[~zeros], signal_counts_weighted[~zeros])
                avg_weight = signal_counts_weighted/signal_counts
                min_variance = avg_weight**2
                replace = (counts <= 0) & (variances < min_variance)
                variances[replace] = min_variance[replace]

        errors = _np.sqrt(variances)

        return counts, errors


    def _Aeff_interpolator(self,filter=None):
        w, a = self['w'], self['a']
        if filter is not None:
            w, a = w[filter], a[filter]
        isort = _np.argsort(w)
        w, a = w[isort], a[isort]
        return lambda wnew: _np.interp(wnew, w, a)


    def _filter_boiler(self, waveranges, time_ranges, order):
        filter = _np.ones(len(self), bool)
        if order is None:
            if 'o' in self and not _np.allclose(self['o'][0], self['o']):
                raise ValueError('You must specify an order number when there are multiple orders in an observation.  '
                                 'Call "merge_orders" if you wish to combine orders.')
        else:
            filter = filter & (self['o'] == order)
        if waveranges is not None:
            filter = filter & utils.inranges(self['w'], waveranges)
        if time_ranges is not None:
            filter = filter & utils.inranges(self['t'], time_ranges)
        return filter

    #endregion


def _groom_bins(bins, rng, bin_method='full'):
    # if bins is a float, parse the full range of observations into bins using that float as the bin width,
    # with unused "leftovers" on each end
    assert bin_method in ['full', 'elastic', 'partial']
    if isinstance(bins, float):
        dx = bins
        x0, x1 = rng
        if (x1 - x0) % dx == 0:
            return  _np.arange(x0, x1+dx, dx)
        elif bin_method == 'elastic':
            n = int(round((x1 - x0) / dx))
            return _np.linspace(x0, x1, n+1)
        else:
            edges_left_aligned =_np.arange(x0, x1, dx)
            edges_full = edges_left_aligned + (x1 - edges_left_aligned[-1])/2.0
            if bin_method == 'full':
                return edges_full
            if bin_method == 'partial':
                return _np.insert(edges_full, [0, len(edges_full)], [edges_full[0]-dx, edges_full[-1]+dx])

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
    Compute an n-point moving sum of the data in vector x. Result will have a length of len(x) - (n-1). Using
    loop avoids the accumulated errors that can result from using numpy.cumsum, though cumsum
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
    xmids = (temp[:-1] + temp[1:]) / 2.0
    bin_start = xmids[:-n]
    bin_stop = xmids[n:]
    return bin_start, bin_stop


def _smooth_boilerplate(x, weights, n, xrange=None, independent=False):
    # first, combine photons with the same time stamp. This isn't ideal, but it prevents errors if n is less than the
    # greatest number of photons with the same stamp
    xu = _np.unique(x)
    if len(xu) < len(x):
        edges = (xu[1:] + xu[:-1]) / 2.0
        edges = _np.insert(edges, [0, len(edges)], [edges[0] - 1, edges[-1] + 1])
        weights, _ = _np.histogram(x, edges, weights=weights)
        x = xu

    counts = _smooth_sum(weights, n)
    errors = _np.sqrt(_smooth_sum(weights**2, n))
    bin_start, bin_stop = _smooth_bins(x, n, xrange)
    assert len(counts) == len(errors) == len(bin_start) == len(bin_stop)

    if independent:
        slc = slice(n//2, None, n)
        bin_start, bin_stop, counts, errors = [a[slc] for a in [bin_start, bin_stop, counts, errors]]

    assert _np.all(bin_start[1:] >= bin_start[:-1])
    assert _np.all(bin_stop[1:] >= bin_stop[:-1])

    # divide by bin wdiths and exposure times to get rates
    bin_widths = bin_stop - bin_start
    rates = counts/bin_widths
    errors = errors/bin_widths

    # bin midpoints
    bin_midpts = (bin_start + bin_stop)/2.0

    return bin_start, bin_stop, bin_midpts, rates, errors


def _inbins(bins, values):
    isort = _np.argsort(bins[:, 0])
    bins = bins[isort, :]
    bin_edges = _np.ravel(bins)
    bin_no = _np.searchsorted(bin_edges, values)
    return bin_no % 2 == 1


class Spectrum(object):
    file_suffix = '.spec'
    table_write_format = 'ascii.ecsv'

    # these prevent recursion with __getattr__, __setattr__ when object is being initialized
    ynames = []
    other_data = {}

    def __init__(self, w, y, err=None, dw='guess', yname='y', other_data=None, references=None, notes=None, wbins=None):
        """

        Parameters
        ----------
        w : wavelength array
        y : y data array (e.g. flux, cross-section)
        err : error array
        dw : bin width array. Can also be 'guess' if you want the widths to be inferred from midpoint spacings,
            but note that this process is ambiguous and it is better to provide explicit dw values if possible.
        yname : str | list
            name for the primary dependent data ('flux', 'x', ...). data will be accessible as an attribute as spec.y or
            spec.y_name. can provide multiple if you want aliases.
        other_data : dict or astropy table
            Additional data associated with points/bins.
        references : list
            References for spectrum.
        notes : list
            Notes on spectrum.
        wbins : array
            Wavelength bin edges. Overrides w and dw.

        """

        if wbins is None:
            if type(dw) is str and dw == 'guess':  # first part avoids comparing array to string
                wbins = utils.wave_edges(w.value) * w.unit
            else:
                dw = dw.to(w.unit)
                wbins = utils.edges_from_mids_diffs(w.value, dw.value) * w.unit
                if not _np.allclose(utils.midpts(wbins).value, w.value):
                    raise ValueError('The bin width (dw) values you provided are not consistent with the wavelength '
                                     'midpoints (w).')
        self.wbins = wbins
        self.y = y
        self.e = self.err = self.error = err
        self.ynames = utils.as_list(yname)
        self.other_data = {} if other_data is None else other_data
        self.references = [] if references is None else references
        self.notes = [] if notes is None else notes

    #region properties
    dw = property(lambda self: _np.diff(self.wbins))
    w = property(lambda self: utils.midpts(self.wbins))
    integral = property(lambda self: _np.sum(self.dw * self.y).decompose())
    #endregion

    #region magic
    def __getattr__(self, key):
        if key in self.ynames:
            return self.y
        elif key in self.other_data:
            return self.other_data[key]
        else:
            raise AttributeError('No {} data associated with this spectrum. Use add_data if you want to add it.'
                                 ''.format(key))

    def __setattr__(self, key, value):
        if key in self.ynames:
            self.__dict__['y'] = value
        elif key in self.other_data:
            od = self.__dict__['other_data'][key] = value
        else:
            self.__dict__[key] = value

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        tbl = self.to_table()
        return tbl.__repr__()

    def __len__(self):
        return len(self.y)
    #endregion

    #region utilities

    #endregion

    #region modification methods
    def add_note(self, note):
        self.notes.append(note)

    def add_data(self, name, data):
        self.other_data[name] = data

    def rebin(self, newbins, other_data_bin_methods='avg'):
        ob = self.wbins.to(newbins.unit).value
        nb = newbins.value

        y = self.y.value
        ynew = _rebin.rebin(nb, ob, y, 'avg') * self.y.unit

        if self.e is None:
            enew = None
        else:
            E = self.e.value * _np.diff(ob)
            V = E ** 2
            Vnew = _rebin.rebin(nb, ob, V, 'sum')
            enew = _np.sqrt(Vnew) / _np.diff(nb) * self.e.unit

        if self.other_data is None:
            other_data_new = None
        else:
            if type(other_data_bin_methods) is str:
                methods = [other_data_bin_methods] * len(self.other_data)
            else:
                methods = other_data_bin_methods
            other_data_new = {}
            for key, method in zip(self.other_data, methods):
                d = self.other_data[key]
                other_data_new[key] = _rebin.rebin(nb, ob, d.value, method) * d.unit

        notes, refs = [_copy.copy(a) for a in [self.notes, self.references]]
        newspec = Spectrum(None, ynew, err=enew, yname=self.ynames, notes=notes, references=refs, wbins=newbins,
                           other_data=other_data_new)
        # this might be bad code, but this allows rebin to be used by subclasses (though they should redefine it if
        # they have extra attributes that should be changed when rebinned, of course)
        # FIXME actually I don't think this even works...
        newspec.__class__ = self.__class__
        return newspec

    def clip(self, wavemin, wavemax):
    #endregion

        keep = (self.wbins[1:] > wavemin) & (self.wbins[:-1] < wavemax)

        y = self.y[keep]

        wbins = self.wbins[:-1][keep]
        wmax = self.wbins[1:][keep][-1]
        wbins = _np.append(wbins.value, wmax.value) * self.wbins.unit
        if wbins[0] < wavemin:
            wbins[0] = wavemin
        if wbins[-1] > wavemax:
            wbins[-1] = wavemax

        if self.other_data is None:
            other_data = None
        else:
            other_data = {}
            for key, a in self.other_data.items():
                other_data[key] = a[keep]

        if self.e is None:
            e = None
        else:
            e = self.e[keep]

        notes, refs = [_copy.copy(a) for a in [self.notes, self.references]]
        return Spectrum(None, y, err=e, yname=self.ynames, notes=notes, references=refs, wbins=wbins,
                        other_data=other_data)

    def rescale(self, factor, yunit=None, wunit=None):
        e = self.e*factor if self.e is not None else None
        y = self.y*factor
        wbins = self.wbins
        if yunit is not None:
            y = y.to(yunit)
        if wunit is not None:
            wbins = wbins.to(wunit)
        return Spectrum(None, y, e, wbins=wbins)
    #endregion

    #region analysis
    def integrate(self, *args):
        if len(args) == 1:
            wbin = args[0]
            try:
                wbin.to(self.w.unit)
            except AttributeError, _u.UnitConversionError:
                raise ValueError('Input must be astropy quantity with units convertable to that of spectrum.')
        elif len(args) == 2:
            w0, w1 = args
            wbin = [w0.value, w1.to(w0.unit).value] * w0.unit
        else:
            raise ValueError('Input must either be [[w0,w1],[w2,w3],...], [w0,w1], or w0, w1.')
        if _np.any(wbin < self.w[0]) or _np.any(wbin > self.w[-1]):
            raise ValueError('Integration range is beyond spectrum range.')
        wbin = wbin.reshape(-1, 2)
        if _np.any(wbin[1:,0] <= wbin[:-1,1]):
            raise ValueError('No overlapping or touching ranges allowed.')
        dw = _np.diff(wbin, 1).squeeze()
        newspec = self.rebin(wbin.ravel())
        flux_dens, err_dens = newspec.y[::2], newspec.e[::2]
        fluxes, errs = flux_dens*dw, err_dens*dw
        flux, error = _np.sum(fluxes), utils.quadsum(errs)
        return flux, error

    def plot(self, *args, **kw):
        ykey = kw.pop('y', 'y')
        ax = kw.pop('ax', _pl.gca())
        draw = kw.pop('draw', True)
        wunit = kw.pop('wunit', None)
        yunit = kw.pop('yunit', None)
        vref = kw.pop('vref', None)
        vunit = kw.pop('vunit', None)

        w, y = [_np.empty(2 * len(self)) for _ in range(2)]
        w[::2], w[1::2] = self.wbins[:-1], self.wbins[1:]
        y[::2] = getattr(self, ykey)
        y[1::2] = y[::2]

        w = w * self.wbins.unit
        y = y * self.y.unit
        if wunit is not None:
            w = w.to(wunit)
        if yunit is not None:
            y = y.to(yunit)

        if vref is None:
            ln = ax.plot(w, y, *args, **kw)
        else:
            v = (w - vref)/vref * _const.c
            if vunit is None:
                v = v.to('km s-1')
            else:
                v = v.to(vunit)
            ln = ax.plot(v, y, *args, **kw)
        if draw:
            _pl.draw()

        return ln
    #endregion

    #region create, import, and export
    def to_table(self):
        data = [self.w, self.dw, self.y]
        names = ['w', 'dw', 'y']
        if self.e is not None:
            data.append(self.e)
            names.append('err')
        if self.other_data is not None:
            for key, val in self.other_data.items():
                data.append(val)
                names.append(key)
        tbl = _table.Table(data=data, names=names)
        tbl.meta['ynames'] = self.ynames
        tbl.meta['notes'] = self.notes
        tbl.meta['references'] = self.references
        return tbl


    def write(self, path, overwrite=False):
        if not path.endswith(Spectrum.file_suffix):
            path += Spectrum.file_suffix

        tbl = self.to_table()
        tbl.write(path, format=Spectrum.table_write_format, overwrite=overwrite)


    @classmethod
    def read_x1d(cls, path, keep_extra_fields=False, keep_header=True):
        h = _fits.open(path)

        std_names = ['wavelength', 'flux', 'error']
        std_data = {}
        other_data = {}
        hdr = h[1].header
        keys = hdr.keys()
        keys = filter(lambda s: 'TTYPE' in s, keys)
        for key in keys:
            name = hdr[key].lower()
            vec = h[1].data[name]
            if vec.ndim < 2 or vec.shape[1] == 1:
                continue
            unit_key = key.replace('TYPE', 'UNIT')
            units_str = hdr[unit_key] if unit_key in hdr else ''
            if units_str == 'Angstroms':
                units_str = 'Angstrom'
            if units_str == 'Counts/s':
                units_str = 'count s-1'
            units = _u.Unit(units_str)
            vec = vec * units
            if name in std_names:
                std_data[name] = vec
            elif keep_extra_fields:
                other_data[name] = vec

        ynames = ['f', 'flux']
        notes = h[0].header + h[1].header if keep_header else None

        specs = []
        for i in range(len(std_data['wavelength'])):
            w, f, e = [std_data[s][i] for s in std_names]
            wbins = utils.wave_edges(w.value) * w.unit
            assert _np.allclose(utils.midpts(wbins.value), w.value)
            if keep_extra_fields:
                other_dict = {}
                for key in other_data:
                    other_dict[key] = other_data[key][i]
            else:
                other_dict = None
            spec = Spectrum(None, f, e, wbins=wbins, notes=notes, other_data=other_dict, yname=ynames)
            specs.append(spec)

        return specs

    @classmethod
    def read_muscles(cls, path):
        h = _fits.open(path)
        w0, w1, f, e = [h[1].data[s] for s in ['w0', 'w1', 'flux', 'error']]
        gaps = w0[1:] != w1[:-1]
        igaps, = _np.nonzero(gaps)
        f, e = [_np.insert(a, igaps, _np.nan) for a in [f, e]]
        fcgs = _u.Unit('erg cm-2 s-1 AA-1')
        wedges = _np.unique(_np.concatenate([w0, w1]))
        return Spectrum(None, f*fcgs, e*fcgs, wbins=wedges*_u.AA, yname=['f', 'flux'])

    @classmethod
    def read(cls, path_or_file_like):
        """
        Read in a spectrum.

        Parameters
        ----------
        path_or_file_like

        Returns
        -------
        Spectrum object
        """
        if type(path_or_file_like) is str and not path_or_file_like.endswith(cls.file_suffix):
            raise IOError('Can only read {} file.'.format(cls.file_suffix))

        tbl = _table.Table.read(path_or_file_like, format='ascii.ecsv')
        w, dw, y = [tbl[s].quantity for s in ['w', 'dw', 'y']]
        tbl.remove_columns(['w', 'dw', 'y'])
        if 'err' in tbl.colnames:
            e = tbl['err']
            tbl.remove_column('e')
        else:
            e = None

        refs = tbl.meta['references']
        notes = tbl.meta['notes']
        ynames = tbl.meta['ynames']

        if len(tbl.colnames) > 0:
            other_data = {}
            for key in tbl.colnames:
                other_data[key] = tbl[key].quantity
        else:
            other_data = None

        spec = Spectrum(w, y, e, dw=dw, other_data=other_data, yname=ynames, references=refs, notes=notes)
        return spec

    @classmethod
    def blackbody(cls, T, wbins):
        # TODO make better by computing integral over bins
        w = utils.midpts(wbins)
        f = _np.pi * 2 * _const.h * _const.c ** 2 / w ** 5 / (_np.exp(_const.h * _const.c / _const.k_B / T / w) - 1)
        f = f.to('erg s-1 cm-2 AA-1')
        return Spectrum(None, f, yname=['f', 'flux', 'surface flux'], wbins=wbins)
    #endregion

    pass