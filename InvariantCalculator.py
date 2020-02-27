import numpy as np


def get_gamma_factor_wrt_rest_frame(p_prime, m):
    return np.sqrt((p_prime / m) ** 2 + 1)


def gamma_to_v(gamma):
    return np.sqrt(1 - gamma**(-2))
