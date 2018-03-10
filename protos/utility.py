import numpy as np
from scipy import interpolate
from fastdtw import fastdtw

# from https://github.com/r9y9/nnmnkwii/blob/4cade86b5c35b4e35615a2a8162ddc638018af0e/nnmnkwii/preprocessing/alignment.py#L14
class DTWAligner(object):
    def __init__(self, dist=lambda x, y: norm(x - y), radius=1, verbose=0):
        self.verbose = verbose
        self.dist = dist
        self.radius = radius

    def transform(self, XY_src, XY_dst=None):
        if XY_dst is None:
            XY_dst = XY_src

        X_src, Y_src = XY_src
        X_dst, Y_dst = XY_dst
        assert X_src.ndim == 3 and Y_src.ndim == 3
        assert X_dst.ndim == 3 and Y_dst.ndim == 3

        longer_features = X_dst if X_dst.shape[1] > Y_dst.shape[1] else Y_dst

        X_aligned = np.zeros_like(longer_features)
        Y_aligned = np.zeros_like(longer_features)
        for idx, (x_src, y_src, x_dst, y_dst) in enumerate(zip(X_src, Y_src, X_dst, Y_dst)):
            dist, path = fastdtw(x_src, y_src, radius=self.radius, dist=self.dist)
            dist /= (len(x_src) + len(y_src))

            pathx = np.array(list(map(lambda l: l[0], path))) / len(x_src)
            pathx = interpolate.interp1d(np.linspace(0, 1, len(pathx)), pathx)(np.linspace(0, 1, len(x_dst)))
            pathx = np.floor(pathx * len(x_dst)).astype(np.int)

            pathy = np.array(list(map(lambda l: l[1], path))) / len(y_src)
            pathy = interpolate.interp1d(np.linspace(0, 1, len(pathy)), pathy)(np.linspace(0, 1, len(y_dst)))
            pathy = np.floor(pathy * len(y_dst)).astype(np.int)

            x_dst, y_dst = x_dst[pathx], y_dst[pathy]
            max_len = max(len(x_dst), len(y_dst))
            if max_len > X_aligned.shape[1] or max_len > Y_aligned.shape[1]:
                pad_size = max(max_len - X_aligned.shape[1],
                               max_len > Y_aligned.shape[1])
                X_aligned = np.pad(
                    X_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
                Y_aligned = np.pad(
                    Y_aligned, [(0, 0), (0, pad_size), (0, 0)],
                    mode="constant", constant_values=0)
            X_aligned[idx][:len(x_dst)] = x_dst
            Y_aligned[idx][:len(y_dst)] = y_dst
            if self.verbose > 0:
                print("{}, distance: {}".format(idx, dist))
        return X_aligned, Y_aligned