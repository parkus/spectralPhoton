import astropy.io.fits as _fits
import astropy.time as _time
import astropy.table as _tbl
import astropy.units as _u
import numpy as _np

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
    - cross dispersion distance, y
    - event weight (termed epsilon for HST observations), e
    - anything else the user want's to define

    Some operators have been defined for this class:
    #FIXME

    These attributes are all protected in this class and are accessed using the bracket syntax, e.g. if `photons` is an
    instance of class `Photons`, then

    >>> photons['w']

    will return the photon wavelengths.

    I thought about deriving this class from FITS_rec or np.recarray, but ultimately it felt cumbersome. I decided
    it's better they be a part of the object, rather than a parent to it, since it will also contain various metadata.
    """

    # for ease of use, map some alternative names to the proper photon property names
    _alternate_names = {'time':'t',
                        'wavelength':'w', 'wave':'w', 'wvln':'w', 'wav':'w', 'waveln':'w',
                        'effective area':'a', 'area':'a',
                        'data quality':'q', 'dq':'q', 'quality':'q', 'flags':'q',
                        'order':'o', 'segment':'o', 'seg':'o',
                        'observation':'n', 'obs':'n',
                        'xdisp':'y', 'cross dispersion':'y',
                        'weight':'e', 'eps':'e', 'epsilon':'e', 'event weight':'e', 'wt':'e'}


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
            self.data = kwargs['data']
            if len(self.obs_metadata) > 1:
                if 'n' not in self.data.colnames:
                    raise ValueError('Column "n" linking photons to observation metadata required if observatio '
                                     'metadata list of len > 1 is supplied.')
                elif self.data['n'].max() > len(self.obs_metadata) - 1:
                    raise ValueError('No photons can be associated with an observation number greater than the number of metadata units in the list.')
        else:
            self.data = _tbl.Table(names=['t', 'w'], dtype=['f8', 'f8'])
            self.data['t'].unit = _u.s
            self.data['w'].unit = _u.AA


    def __getitem__(self, key):
        key = self._get_proper_key(key)
        return self.data[key]


    def __setitem__(self, key, value):
        key = self._get_proper_key(key)
        self.data[key] = value


    def _get_proper_key(self, key):
        key = key.lower()
        if key in self.field_dict.values():
            return key
        elif key in self.field_dict:
            return self.field_dict[key]
        else:
            raise KeyError('{} not recognized as a field name'.format(key))


    def copy(self):
        new = Photons()
        new.obs_metadata = [item.copy() for item in self.obs_metadata]
        new.time_datum = self.time_datum
        new.data = self.data.copy()


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
        for key in other.colnames:
            if key in self.colnames:
                if self[key].unit:
                    unit = self[key].unit
                    other[key].convert_unit_to(unit)

        # add and /or update observation columns as necessary
        if 'n' not in self.data.colnames:
            n_ary = _np.zeros(len(self.data))
            n_col = _tbl.Column(data=n_ary, dtype='i2', name='n')
            self.data.add_column(n_col)
        n_obs_self = len(self.obs_metadata)
        if 'n' in other.data.colnames:
            other['n'] += n_obs_self
        else:
            n_ary = _np.ones(len(self.data)) + n_obs_self
            n_col = _tbl.Column(data=_np.zeros(len(self.data)), dtype='i2', name='n')
            other.data.add_column(n_col)

        # re-reference times to the datum of self
        other.set_time_datum(self.time_datum)

        # stack the data tables
        data = _tbl.vstack([self, other])

        # leave it to the user to deal with sorting and grouping and dealing with overlap as they see fit

        obs_metadata = self.obs_metadata + other.obs_metadata

        sum = Photons()
        sum.data =





