import numpy as _np

def wave_edges(wave_midpts):
    """Reconstructs the wavelength bins used in the x1d.

    Assumes a linear change in the wavelength step."""
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
    # convert bin density to bin value
    lo, hi = bin_edges[:-1], bin_edges[1:]
    d = _np.diff(bin_edges)
    y, e = density*d, error*d
    v = e**2 # variance

    # iteratively add pairs of bins with their neighbors (alternating which neighbor)
    forward = True
    while True:
        # identify the low S/N points. break if there aren't any
        low_SN = y/e < min_SN
        if not _np.any(low_SN) or len(y) <= 1:
            break
        i_low_SN, = _np.where(low_SN)

        # remove any adjacent low S/N bins or else the flux will be double-counted using the vector operations below
        adjacent = _np.zeros_like(i_low_SN, bool)
        adjacent[1:] = (i_low_SN[1:] == i_low_SN[:-1] + 1)
        # however, we need only worry about every other adjacent bin
        adjacent[::2] = False
        i_low_SN = i_low_SN[~adjacent]

        # now sum pairs of bins
        if forward:
            # if the last pt is negative, forget it for this iteration
            if i_low_SN[-1] == len(y) - 1:
                i_low_SN = i_low_SN[:-1]
            sum_i = i_low_SN + 1
            hi[i_low_SN] = hi[sum_i]
        else:
            # now its the first point we need to worry about
            if i_low_SN[0] == 0:
                i_low_SN = i_low_SN[1:]
            sum_i = i_low_SN - 1
            lo[i_low_SN] = lo[sum_i]
        y[i_low_SN] += y[sum_i]
        v[i_low_SN] += v[sum_i]

        # and delete one of the pair that have been summed
        lo, hi, y, v, e = [_np.delete(a, sum_i) for a in [lo, hi, y, v, e]]

        # and flip the forward switch so next iteration neighbors to the other side will be used
        forward = not forward

    # convert back to density (ds means downsampled)
    d_ds = hi - lo
    density_ds, error_ds = y/d_ds, _np.sqrt(v)/d_ds
    bin_edges_ds = _np.append(lo, hi[-1])
    return bin_edges_ds, density_ds, error_ds