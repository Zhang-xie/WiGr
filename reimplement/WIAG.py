import numpy as np
from utils import pca_matlab, gen_iir_filter
from scipy.interpolate import UnivariateSpline
import pywt
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier

b_filter = gen_iir_filter(1000)
N_tx = 2
N_rx = 3
subc_n = 30
data_n = 128
class_num = 6


def _f(K, Di, d, thetai, THETA):
    return K * Di ** 2 * (1 + (d / Di) ** 2 - 2 * (d / Di) * np.cos(thetai + THETA))


# def translation_func(xA):
#     xB = xA * _f(KB, DiA, d, thetai_A, THETA_A) / _f(KA, DiB, d, thetai_B, THETA_B)


def translation_func_for_speed_test(xA):
    t = xA.shape[0]
    KA = np.random.rand()
    DiA = np.random.rand(t)[:, None]
    thetai_A = np.random.rand(t)[:, None]
    THETA_A = np.random.rand()

    KB = np.random.rand()
    DiB = np.random.rand(t)[:, None]
    thetai_B = np.random.rand(t)[:, None]
    THETA_B = np.random.rand()

    d = np.random.rand()

    xB = xA * _f(KB, DiA, d, thetai_A, THETA_A) / _f(KA, DiB, d, thetai_B, THETA_B)
    return xB


def fit_config(x):
    res = Polynomial.fit(np.arange(len(x)), x, 8)
    a0, a1, a2 = res.coef[:3]
    v = np.random.rand()

    def equations(vars):
        D, THETA, k = vars
        return (
            k * v / D ** 2 - a0,
            a0 * (2 * (v / D) * np.cos(THETA)) - a1,
            -a0 * ((v / D) ** 2 * (1 - 4 * np.cos(THETA) ** 2)) - a2,
        )

    D, THETA, k = fsolve(equations, (1, 1, 1))
    return D, THETA, k


def gen_dPC_streams(csi):
    dPC = []
    # dPC_eigenV = []
    # dPC_eigenV.append(latent[2])
    for i in range(N_tx):
        for j in range(N_rx):
            coeff, score, latent = pca_matlab(csi[:, i, j, :])
            _PC = np.abs(score[:, 2])
            # _PC_filted = scipy.ndimage.gaussian_filter(_PC, sigma=(3,))
            # _PC_filted = scipy.ndimage.uniform_filter(PC, size=(25,))
            _PC_filted = b_filter(x=_PC)
            _dPC = np.diff(_PC_filted, n=1)
            dPC.append(_dPC)
    return np.stack(dPC, axis=1)


def extract_feature(dPC):

    # dPC = np.array(dPC).T
    coeff, score, latent = pca_matlab(dPC)

    dPC_c = score[:, 0]

    spl = UnivariateSpline(np.linspace(0, 100, len(dPC_c)), dPC_c)
    spl.set_smoothing_factor(0.0)
    dPC_c_extra = spl(np.linspace(0, 100, 1024))

    db = pywt.Wavelet("db2")
    _, dPC_c_dwt, _, _ = pywt.wavedec(dPC_c_extra, db, mode="periodization", level=3)

    dPC_c_bin = np.split(dPC_c_dwt, [13, 26, 39, 52, 65, 78, 91, 104, 116])
    dPC_c_feature = np.array([np.sum(v ** 2) for v in dPC_c_bin])
    return dPC_c_feature


data = [np.random.rand(900, 2, 3, subc_n) for i in range(data_n)]


# STEP1 CSI-Stream Conditioning
start = perf_counter()
dPC = [gen_dPC_streams(x) for x in data]
t1 = perf_counter() - start

# STEP2 Configuration Estimation
config = [
    [fit_config(dPC[i][:, j]) for j in range(N_rx * N_tx)] for i in range(len(dPC))
]
t2 = perf_counter() - t1
# STEP3 Gesture Translation
dPC_b = [translation_func_for_speed_test(x) for x in dPC]
t3 = perf_counter() - t2
# STEP4 Classifier
x = np.array([extract_feature(x) for x in dPC + dPC_b])
y = np.random.randint(0, class_num, data_n * 2)
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(x, y)
t4 = perf_counter() - t3
y_predict = neigh.predict(x)
t5 = perf_counter() - t4
print(t1 / data_n, t2 / data_n, t3 / data_n, t4 / data_n / 2, t5 / data_n / 2)
