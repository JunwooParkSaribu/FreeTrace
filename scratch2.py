from stochastic.processes.noise import FractionalGaussianNoise
import numpy as np

rng = np.random.default_rng(1)
a = np.cumsum(FractionalGaussianNoise(hurst=0.999, t=1, rng=rng).sample(10, algorithm='hosking'))
print(a)