from numpy import clip, load, ndarray, pi, sign, sin
from numpy.random import normal, uniform
from os.path import dirname, realpath


def small_change(u: ndarray, u_min: ndarray, u_max: ndarray, u_severity: float, **kwargs) -> ndarray:
    alpha = 0.04
    u_range = u_max-u_min
    r = uniform(-1, 1) if isinstance(u, (int,float)) else uniform(-1, 1, u.shape)
    u += alpha * u_range * r * u_severity
    u = clip(u, u_min, u_max)
    return u


def large_change(u: ndarray, u_min: ndarray, u_max: ndarray, u_severity: float, **kwargs) -> ndarray:
    alpha = 0.04
    alpha_max = 0.1
    u_range = u_max-u_min
    r = uniform(-1, 1) if isinstance(u, (int,float)) else uniform(-1, 1, u.shape)
    u += u_range * (alpha*sign(r)+(alpha_max-alpha)*r) * u_severity
    u = clip(u, u_min, u_max)
    return u


def random_change(u: ndarray, u_min: ndarray, u_max: ndarray, u_severity: float, **kwargs) -> ndarray:
    u += normal(0, u_severity) if isinstance(u, (int,float)) else normal(0, u_severity, u.shape)
    u = clip(u, u_min, u_max)
    return u


def chaotic_change(u: ndarray, u_min: ndarray, u_max: ndarray, **kwargs) -> ndarray:
    A = 3.67
    u_range = u_max-u_min
    u = (u-u_min) / u_range
    u = A * u * (1-u)
    u = u_min + u * u_range
    return u


def recurrent_change(u: ndarray, u_min: ndarray, u_max: ndarray, change_count: int, **kwargs) -> ndarray:
    p = 12
    u_range = u_max-u_min
    phi = load(f"{dirname(realpath(__file__))}/dat/phi.npy")
    phi = phi[0] if isinstance(u, (int,float)) else phi[:u.shape[0]]
    u = u_min + u_range * (sin(2*pi*change_count/p+phi)+1) / 2
    return u


def noisy_recurrent_change(u: ndarray, u_min: ndarray, u_max: ndarray, change_count: int, **kwargs) -> ndarray:
    p = 12
    noisy_severity = 0.8
    u_range = u_max-u_min
    phi = load(f"{dirname(realpath(__file__))}/dat/phi.npy")
    phi = phi[0] if isinstance(u, (int,float)) else phi[:u.shape[0]]
    err = normal(0, noisy_severity) if isinstance(u, (int,float)) else normal(0, noisy_severity, u.shape)
    u = u_min + u_range*(sin(2*pi*change_count/p+phi)+1) / 2 + err
    u = clip(u, u_min, u_max)
    return u
