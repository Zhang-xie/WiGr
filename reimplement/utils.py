import functools

import scipy.io
import scipy.signal
import scipy.linalg
import numpy as np
from time import perf_counter


def pca_matlab(X):
    def complex_sign(complex):
        return complex / abs(complex)

    center = X - X.mean(axis=0)
    U, S, V = scipy.linalg.svd(center, full_matrices=False)

    # score = U * S
    latent = S ** 2 / (X.shape[0] - 1)

    V = V.conj().T  #

    # flip eigenvectors' sign to enforce deterministic output
    max_abs_idx = np.argmax(abs(V), axis=0)
    colsign = complex_sign(
        V[
            max_abs_idx,
            range(V.shape[0]),
        ]
    )
    U *= colsign
    V *= colsign[None, :]
    score = U * S
    # above code makes: center == U @ np.diag(S) @ V.conj().T
    coeff = V
    return coeff, score, latent


def gen_iir_filter(sample_rate):
    samp_rate = sample_rate
    half_rate = samp_rate / 2
    uppe_stop = 40
    [lb, la] = scipy.signal.butter(6, uppe_stop / half_rate, "lowpass")

    return functools.partial(scipy.signal.filtfilt, b=lb, a=la, axis=0, method="gust")


def get_nn_running_time(func, loop_num=10):
    for i in range(5):
        func()

    start_t = perf_counter()
    for i in range(loop_num):
        func()
    t = (perf_counter() - start_t) / loop_num
    return t
