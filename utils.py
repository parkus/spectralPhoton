import numpy as _np


class LowSNError(BaseException):
    pass


def wave_edges(wave_midpts):
    """
    Reconstructs the wavelength bins used in the x1d.

    Assumes a linear change in the wavelength step.

    Parameters
    ----------
    wave_midpts

    Returns
    -------
    edges
    """
    w = wave_midpts

    # apply a qudratic fit to the bin widths to estimate the bin width at the first point
    mid_mid = (w[:-1] + w[1:])/2.0 # midpts of the midpts :)
    dw = w[1:] - w[:-1]
    fit_coeffs = _np.polyfit(mid_mid, dw, 2)

    # now start with the left edge and propagate
    dw_left_end = _np.polyval(fit_coeffs, w[0])
    edges = [w[0] - dw_left_end/2.0]
    for ww in w:
        edges.append(2*ww - edges[-1])

    # this whole shebang might have seemed silly, but I have tried endless variations of "meet in the middle"
    # approaches to trying to get the best possible bin edges and none are great. this should be fine

    return _np.array(edges)


def edges_from_mids_diffs(w, dw):
    return _np.append(w - dw/2., w[-1])


def edges_from_mids_coarse(w):
    mid_mid = (w[:-1] + w[1:])/2.0
    dwa = w[1] - w[0]
    dwb = w[-1] - w[-2]
    edges = _np.insert(mid_mid, (0, len(mid_mid)),
                       (mid_mid[0] - dwa/2, mid_mid[-1] + dwb/2))
    return edges


def as_list(a):
    if a is None:
        return None
    elif type(a) is not list:
        return [a]
    else:
        return a


def adaptive_downsample(bin_edges, density, error, min_SN):
    """

    Parameters
    ----------
    bin_edges
    density
    error
    min_SN

    Returns
    -------
    bin_edges_ds, density_ds, error_ds
    """

    bin_edges, density, error = map(_np.asarray, [bin_edges, density, error])

    # assume multiple spectra are present. if not reshape two have two dimensions
    if density.ndim == 1:
        bin_edges, density, error = [_np.reshape(a, [1, None]) for a in bin_edges, density, error]
    else:
        assert len(density) == len(error)
        if bin_edges.ndim > 1:
            raise NotImplementedError('Can\'t handle different bin edges for each spectrum/ If they\'re all the same, '
                                      'just give a 1D array specifiying the bin edges for each spectrum.')

    # convert bin density to bin value
    lo, hi = bin_edges[:-1].copy(), bin_edges[1:].copy()
    d = _np.diff(bin_edges)
    y, e = density*d[_np.newaxis, :], error*d[_np.newaxis, :]
    v = e**2 # variance

    # check that the spectra has sufficient S/N to complete the process
    I = _np.sum(y*d[_np.newaxis,:], axis=1)
    E = _np.sqrt(_np.sum((e*d[_np.newaxis,:])**2, axis=1))
    if any(I/E < min_SN):
        raise LowSNError('At least one spectrum does not have an integrated S/N exceeding the min_SN setting of {}.'
                         ''.format(min_SN))

    # iteratively add pairs of bins with their neighbors (alternating which neighbor)
    forward = True
    while True:
        # identify the low S/N points. break if there aren't any
        low_SN = y/e < min_SN
        if not _np.any(low_SN) or len(y) <= 1:
            break
        low_SN_cols = reduce(_np.logical_or, low_SN, _np.zeros(low_SN.shape[1], bool))
        i_low_SN, = _np.where(low_SN_cols)

        # remove any adjacent low S/N bins or else the flux will be double-counted using the vector operations below
        adjacent = _np.zeros_like(i_low_SN, bool)
        adjacent[1:] = (i_low_SN[1:] == i_low_SN[:-1] + 1)
        # however, we need only worry about every other adjacent bin
        # thought about doing adjacent[::2] = False, but this causes problems
        i_low_SN = i_low_SN[~adjacent]

        # now sum pairs of bins
        if forward:
            # if the last pt is flagged, forget it for this iteration
            if i_low_SN[-1] == y.shape[1] - 1:
                i_low_SN = i_low_SN[:-1]
            sum_i = i_low_SN + 1
            hi[i_low_SN] = hi[sum_i]
        else:
            # now its the first point we need to worry about
            if i_low_SN[0] == 0:
                i_low_SN = i_low_SN[1:]
            sum_i = i_low_SN - 1
            lo[i_low_SN] = lo[sum_i]
        y[:, i_low_SN] += y[:, sum_i]
        v[:, i_low_SN] += v[:, sum_i]

        # and delete one of the pair that have been summed
        lo, hi = [_np.delete(a, sum_i) for a in [lo, hi]]
        y, v = [_np.delete(a, sum_i, axis=1) for a in [y, v]]
        e = _np.sqrt(v)

        # and flip the forward switch so next iteration neighbors to the other side will be used
        forward = not forward

    # convert back to density (ds means downsampled)
    d_ds = hi - lo
    density_ds, error_ds = y/d_ds, _np.sqrt(v)/d_ds
    bin_edges_ds = _np.append(lo, hi[-1])

    # remove extra dimensions (i.e. if only a single spectrum was used)
    density_ds, error_ds = map(_np.squeeze, [density_ds, error_ds])

    return bin_edges_ds, density_ds, error_ds


def rebin(new_edges, old_edges, density, error=None):

    # get overlapping bins, warn if some don't overlap
    if (new_edges[0] < old_edges[0]) or (new_edges[-1] > old_edges[-1]):
        raise ValueError('New bin edges must fall within old bin edges.')

    dold = old_edges[1:] - old_edges[:-1]
    dnew = new_edges[1:] - new_edges[:-1]

    # making a function so I can easily reuse commands for flux and error
    def compute(z):
        # use interpolation of cumulative integral (as sum of density*dw -- like a Riemann sum) as sneaky and fast way
        # to rebin
        ## get the value of the integral at each old edge for values and variance
        Iz = _np.zeros(len(z) + 1) # first pt needs to be zero, so may as well preallocate
        Iz[1:] = _np.cumsum(z)

        ## interpolate to new edges
        Iz_new = _np.interp(new_edges, old_edges, Iz)

        ## difference of the integral at each edge gives the values in the new bins
        z_new = _np.diff(Iz_new)

        return z_new

    # compute rectangular areas
    y = density*dold

    # sometimes cumsum can result in significant numerical error if summing lots of small numbers, so I'll normalize
    # to be safe
    normfac = _np.nanmedian(y[y > 0])
    normy = y/normfac

    # do the actual rebinning
    normy_new = compute(normy)

    # un-normalize and return to density
    density_new = normy_new*normfac/dnew

    # do all this for errors, if provided
    if error is not None:
        norme = error*dold/normfac
        normv = norme**2 # variance
        normv_new = compute(normv)
        error_new = _np.sqrt(normv_new)*normfac/dnew
        return density_new, error_new
    else:
        return density_new


def rangeset_intersect(ranges0, ranges1, presorted=False):
    """
    Return the intersection of two sets of sorted ranges, given as Nx2 array-like.
    """

    if len(ranges0) == 0 or len(ranges1) == 0:
        return _np.empty([0, 2])
    rng0, rng1 = map(_np.asarray, [ranges0, ranges1])

    if not presorted:
        rng0, rng1 = [r[_np.argsort(r[:,0])] for r in [rng0, rng1]]
    for rng in [rng0, rng1]:
        assert _np.all(rng[:,1] > rng[:,0])

    l0, r0 = rng0.T
    l1, r1 = rng1.T
    f0, f1 = [rng.flatten() for rng in [rng0, rng1]]

    lin0 = inranges(l0, f1, [1, 0])
    rin0 = inranges(r0, f1, [0, 1])
    lin1 = inranges(l1, f0, [0, 0])
    rin1 = inranges(r1, f0, [0, 0])

    #keep only those edges that are within a good area of the other range
    l = weave(l0[lin0], l1[lin1])
    r = weave(r0[rin0], r1[rin1])
    return _np.array([l, r]).T


def rangeset_invert(ranges):
    edges = ranges.ravel()
    rnglist = [edges[1:-1].reshape([-1, 2])]
    if edges[0] != -_np.inf:
        firstrng = [[-_np.inf, edges[0]]]
        rnglist.insert(0, firstrng)
    if edges[-1] != _np.inf:
        lastrng = [[edges[-1], _np.inf]]
        rnglist.append(lastrng)
    return _np.vstack(rnglist)


def rangeset_union(ranges0, ranges1):
    invrng0, invrng1 = map(rangeset_invert, [ranges0, ranges1])
    xinv = rangeset_intersect(invrng0, invrng1)
    return rangeset_invert(xinv)


def rangeset_subtract(baseranges, subranges):
    """Subtract subranges from baseranges, given as Nx2 arrays."""
    return rangeset_intersect(baseranges, rangeset_invert(subranges))


def weave(a, b):
    """
    Insert values from b into a in a way that maintains their order. Both must
    be sorted.
    """
    mapba = _np.searchsorted(a, b)
    return _np.insert(a, mapba, b)


def inranges(values, ranges, inclusive=[False, True]):
    """Determines whether values are in the supplied list of sorted ranges.

    Parameters
    ----------
    values : 1-D array-like
        The values to be checked.
    ranges : 1-D or 2-D array-like
        The ranges used to check whether values are in or out.
        If 2-D, ranges should have dimensions Nx2, where N is the number of
        ranges. If 1-D, it should have length 2N. A 2xN array may be used, but
        note that it will be assumed to be Nx2 if N == 2.
    inclusive : length 2 list of booleans
        Whether to treat bounds as inclusive. Because it is the default
        behavior of numpy.searchsorted, [False, True] is the default here as
        well. Using [False, False] or [True, True] will require roughly triple
        computation time.

    Returns a boolean array indexing the values that are in the ranges.
    """
    if ranges is None:
        return _np.ones_like(values, bool)
    ranges = _np.asarray(ranges)
    if ranges.ndim == 2:
        if ranges.shape[1] != 2:
            ranges = ranges.T
        ranges = ranges.ravel()

    if inclusive == [0, 1]:
        return (_np.searchsorted(ranges, values) % 2 == 1)
    if inclusive == [1, 0]:
        return (_np.searchsorted(ranges, values, side='right') % 2 == 1)
    if inclusive == [1, 1]:
        a = (_np.searchsorted(ranges, values) % 2 == 1)
        b = (_np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a | b)
    if inclusive == [0, 0]:
        a = (_np.searchsorted(ranges, values) % 2 == 1)
        b = (_np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a & b)


def midpts(x):
    return (x[1:] + x[:-1])/2.0


def quadsum(x):
    return _np.sqrt(_np.sum(x**2))


def polyfit_binned(bins, y, yerr, order):
    """Generates a function for the maximum likelihood fit to a set of binned
    data.

    Parameters
    ----------
    bins : 2D array-like, shape Nx2
        Bin edges where bins[0] gives the left edges and bins[1] the right.
    y : 1D array-like, length N
        data, the integral of some value over the bins
    yerr : 1D array-like, length N
        errors on the data
    order : int
        the order of the polynomial to fit

    Returns
    -------
    coeffs : 1D array
        coefficients of the polynomial, highest power first (such that it
        may be used with numpy.polyval)
    covar : 2D array
        covariance matrix for the polynomial coefficients
    fitfunc : function
        Function that evaluates y and yerr when given a new bin using the
        maximum likelihood model fit.
    """
    N, M = order, len(y)
    if type(yerr) in [int,float]: yerr = yerr*np.ones(M)
    bins = _np.asarray(bins)
    assert not _np.any(yerr == 0.0)

    #some prelim calcs. all matrices are (N+1)xM
    def prelim(bins, M):
        a, b = bins[:,0], bins[:,1]
        apow = _np.array([a**(n+1) for n in range(N+1)])
        bpow = _np.array([b**(n+1) for n in range(N+1)])
        bap = bpow - apow
        frac = _np.array([_np.ones(M)/(n+1) for n in range(N+1)])
        return bap, frac

    bap, frac = prelim(bins, M)
    var = _np.array([_np.array(yerr)**2]*(N+1))
    ymat = _np.array([y]*(N+1))

    #build the RHS vector
    rhs = _np.sum(ymat*bap/var, 1)

    #build the LHS coefficient matrix
    nmat = bap/var #N+1xM (n,m)
    kmat = bap.T*frac.T #MxN+1 (m,k)
    lhs = _np.dot(nmat,kmat)

    #solve for the polynomial coefficients
    c = _np.linalg.solve(lhs, rhs)

    #compute the inverse covariance matrix (same as Hessian)
    H = _np.dot(nmat*frac,kmat)
    cov = _np.linalg.inv(H)

    #construct the function to compute model values and errors
    def f(bins):
        M = len(bins)
        bap, frac = prelim(bins, M)

        #values
        cmat = _np.transpose([c]*M)
        y = _np.sum(bap*cmat*frac, 0)

        #errors
        T = bap*frac
        cT = _np.dot(cov, T)
        #to avoid memory overflow, compute diagonal elements directly instead
        #of dotting the matrices
        yvar = _np.sum(T*cT, 0)
        yerr = _np.sqrt(yvar)

        return y, yerr

    return c[::-1], cov[::-1,::-1], f