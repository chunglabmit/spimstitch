import numpy as np

def weighted_median(data, weights):
    """

    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    if len(data) == 0:
        return np.NaN
    data, weights = np.array(data), np.array(weights)
    if len(data) == 1:
        return data[0]
    order = np.argsort(data)
    s_data, s_weights = data[order], weights[order]
    cum_weights = np.cumsum(s_weights)
    frac_weights = cum_weights / cum_weights[-1]
    weights_lt_half = np.sum(frac_weights < .5)
    if weights_lt_half == 0:
        # The lowest has more than .5 of the total weight
        return s_data[0]
    elif weights_lt_half == len(s_data) - 1:
        # The highest has more than .5 of the total weight
        return s_data[-1]
    low = .5 - frac_weights[weights_lt_half - 1] + np.finfo(np.float32).eps
    hi = frac_weights[weights_lt_half] - .5 + np.finfo(np.float32).eps
    low_frac = hi / (low + hi)
    hi_frac = low / (low + hi)
    return low_frac * data[weights_lt_half - 1] + hi_frac * data[weights_lt_half]