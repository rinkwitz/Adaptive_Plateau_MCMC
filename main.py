import numpy as np
from Utils import *

M = 5
N = 100

x = np.random.randn(4)
for n in range(N):
    for k in range(len(x)):
        # 4
        z = [rejection_sampling_trial(x[k], j, M) for j in range(M)]

        # 5

        print(z)
        for j in range(M):
            xs = np.linspace(x[k] - 30, x[k] + 30, 1000)
            ys = [trial_proposal(i, x[k], j, M) for i in xs]
            z = [rejection_sampling_trial(x[k], j, M) for j2 in range(20)]
            plt.plot(xs, ys)
            plt.scatter(z, np.zeros(len(z)))
            print(z)
        plt.show()
        break