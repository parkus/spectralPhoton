import numpy as _np

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


