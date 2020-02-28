import numpy as np
from Utils import *

M = 10
N = 100

x = np.random.randn(4)
for n in range(N):
    for k in range(len(x)):
        # 4
        z = [rejection_sampling_trial(x[k], j, M) for j in range(M)]

        # 5

        print(z)