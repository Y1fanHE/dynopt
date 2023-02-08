from numpy import cos, eye, hstack, load, max, ndarray, ones, pi, sin, sqrt, sum
from numpy.random import permutation, uniform
from os.path import dirname, realpath
from typing import Optional, Sequence
from dynopt.dynamic_change import small_change, large_change, random_change, \
    chaotic_change, recurrent_change, noisy_recurrent_change


class RotationPeak:

    def __init__(
        self,
        dim: int = 10,                              # dimension
        num_peaks: int = 10,                        # number of peaks
        bounds: Sequence[float] = (-5,5),           # search range
        freq: Optional[int] = None,                 # frequency of changes
        num_change: int = 60,                       # total number of changes
        initial_height: float = 50,                 # initial height
        h_min: float = 10,                          # minimum height
        h_max: float = 100,                         # maximum height
        h_severity: float = 5,                      # height severity
        w_severity: float = 0.5,                    # width severity
        initial_width: float = 5,                   # initial width
        w_min: float = 1,                           # minimum width
        w_max: float = 10                           # maximum width
    ):

        self.dim = dim
        self.num_peaks = num_peaks
        self.bounds = bounds
        self.freq = freq
        if not self.freq:
            self.freq = 10000 * self.dim
        self.num_change = num_change
        self.initial_height = initial_height
        self.h_min = h_min
        self.h_max = h_max
        self.h_severity = h_severity
        self.w_severity = w_severity
        self.initial_width = initial_width
        self.w_min = w_min
        self.w_max = w_max

        self.FES = 0
        self.npy_path = dirname(realpath(__file__))

    def evaluate(self, x: ndarray):
        if self.FES == 0:
            self._initialize()
        if self.FES == self.FES_last_change + self.freq:
            self._change()
        f = self._func(x)
        self.FES += 1
        return f, x

    def _func(self, x):
        return max(
            self.h / (
                1 + self.w * sqrt(sum((x-self.x_peaks)**2,axis=1) / self.dim)
            )
        )

    def _initialize(self):
        self.change_count = 0
        self.FES_last_change = 0
        x_peaks_original = load(f"{self.npy_path}/dat/x_peaks_original.npy")
        self.x_peaks = x_peaks_original[:self.num_peaks,:self.dim]

    def _change(self):
        self.change_count += 1
        self.FES_last_change = self.FES

    def _perm_dim(self):
        l = (self.dim-1) * (self.dim%2) + self.dim * (1-(self.dim%2))
        r = permutation(self.dim)
        return r[:l]

    def _rotate_peaks(self, r):
        rotation_matrix = eye(self.dim)
        for i in range(0, len(r)//2):
            rotation_matrix[r[i*2],r[i*2]] = cos(self.theta)
            rotation_matrix[r[i*2+1],r[i*2+1]] = cos(self.theta)
            rotation_matrix[r[i*2],r[i*2+1]] = -sin(self.theta)
            rotation_matrix[r[i*2+1],r[i*2]] = sin(self.theta)
        return self.x_peaks @ rotation_matrix


class SmallRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.theta = uniform(-pi, pi)

    def _change(self):
        super()._change()
        self.h = small_change(self.h, self.h_min, self.h_max, self.h_severity)
        self.w = small_change(self.w, self.w_min, self.w_max, self.w_severity)
        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = small_change(self.theta, -pi, pi, 1)


class LargeRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.theta = uniform(-pi, pi)

    def _change(self):
        super()._change()
        self.h = large_change(self.h, self.h_min, self.h_max, self.h_severity)
        self.w = large_change(self.w, self.w_min, self.w_max, self.w_severity)
        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = large_change(self.theta, -pi, pi, 1)


class RandomRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.theta = uniform(-pi, pi)

    def _change(self):
        super()._change()
        self.h = random_change(self.h, self.h_min, self.h_max, self.h_severity)
        self.w = random_change(self.w, self.w_min, self.w_max, self.w_severity)
        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = random_change(self.theta, -pi, pi, 1)


class ChaoticRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        h_original = load(f"{self.npy_path}/dat/h_original.npy")
        w_original = load(f"{self.npy_path}/dat/w_original.npy")
        self.h = h_original[:self.num_peaks]
        self.w = w_original[:self.num_peaks]

    def _change(self):
        super()._change()
        self.h = chaotic_change(self.h, self.h_min, self.h_max)
        self.w = chaotic_change(self.w, self.w_min, self.w_max)
        self.x_peaks = chaotic_change(self.x_peaks, *self.bounds)


class RecurrentRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.h = recurrent_change(self.h, self.h_min, self.h_max, 0)
        self.w = recurrent_change(self.w, self.w_min, self.w_max, 0)
        self.theta = recurrent_change(0, 0, pi/6, 1)

    def _change(self):
        super()._change()
        self.h = recurrent_change(self.h, self.h_min, self.h_max, self.change_count)
        self.w = recurrent_change(self.w, self.w_min, self.w_max, self.change_count)
        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = recurrent_change(self.theta, 0, pi/6, self.change_count)


class NoisyRecurrentRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.h = noisy_recurrent_change(self.h, self.h_min, self.h_max, 0)
        self.w = noisy_recurrent_change(self.w, self.w_min, self.w_max, 0)
        self.theta = noisy_recurrent_change(0, 0, pi/6, 1)

    def _change(self):
        super()._change()
        self.h = noisy_recurrent_change(self.h, self.h_min, self.h_max, self.change_count)
        self.w = noisy_recurrent_change(self.w, self.w_min, self.w_max, self.change_count)
        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = noisy_recurrent_change(self.theta, 0, pi/6, self.change_count)


class RandomDimensionalRotationPeak(RotationPeak):

    def __init__(self, dim: int = 10, num_peaks: int = 10):
        super().__init__(dim, num_peaks)

    def evaluate(self, x: ndarray):
        if self.FES == 0:
            self._initialize()
        if self.FES == self.FES_last_change + self.freq:
            self._change()
            x = self._fix_x_dim(x)
        f = self._func(x)
        self.FES += 1
        return f, x

    def _initialize(self):
        super()._initialize()
        self.h = self.initial_height * ones((self.num_peaks,1))
        self.w = self.initial_width * ones((self.num_peaks,1))
        self.dim_change = 1
        self.theta = uniform(-pi, pi)

    def _change(self):
        super()._change()
        self.h = random_change(self.h, self.h_min, self.h_max, self.h_severity)
        self.w = random_change(self.w, self.w_min, self.w_max, self.w_severity)

        if self.dim+self.dim_change > 15 or self.dim+self.dim_change < 5:
            self.dim_change *= -1
        self.dim += self.dim_change
        if self.dim_change > 0:   # Dimension is increased by 1.
            x_peaks_original = load(f"{self.npy_path}/dat/x_peaks_original.npy")
            self.x_peaks = hstack([self.x_peaks, x_peaks_original[:self.num_peaks,self.dim:self.dim+1]])
        else:                     # Dimension is decreased by 1.
            self.x_peaks = self.x_peaks[:,:self.dim]

        r = self._perm_dim()
        self.x_peaks = self._rotate_peaks(r)
        self.theta = random_change(self.theta, -pi, pi, 1)

    def _fix_x_dim(self, x):
        if self.dim_change > 0:   # Dimension is increased by 1.
            x = hstack([x, uniform(*self.bounds)])
        else:                     # Dimension is decreased by 1.
            x = x[:self.dim]
        return x
