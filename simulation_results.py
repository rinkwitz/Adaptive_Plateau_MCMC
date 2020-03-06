import numpy as np
import matplotlib.pyplot as plt
from Utils import *

M = 5
N = 1000
target_dist = 'pi_test'
delta = 2.
delta_1 = 2.
sigma = .05
sigma_0 = 3.
sigma_1 = 3.

dim_dict = {'pi_test': 1,
            'pi_1': 4,
            'pi_2': 8,
            'pi_3': 2,
            'pi_4': 1}

# algorithm 1, line 1
x = np.random.randn(dim_dict[target_dist])
X = [x]

# algorithm 1, line 2
for n in range(N):
    print(n, x)

    # algorithm 1, line 3
    for k in range(len(x)):

        # algorithm 1, line 4
        z = [rejection_sampling_trial(x[k], j, M, delta, delta_1, sigma, sigma_0, sigma_1) for j in range(M)]

        # algorithm 1, line 5
        w = [trial_weight(z[j], x, k, j, M, target_dist, delta, delta_1, sigma, sigma_0, sigma_1) for j in range(M)]

        # algorithm 1, line 6
        y = draw_from_z_proportional_to_w(z, w)

        # algorithm 1, line 7
        x_star = [rejection_sampling_trial(y, j, M, delta, delta_1, sigma, sigma_0, sigma_1) for j in range(M - 1)]
        x_star.append(x[k])

        # algorithm 1, line 8
        y_bold = x.copy()
        y_bold[k] = y
        num = np.sum([trial_weight(z[j], x, k, j, M, target_dist, delta, delta_1, sigma, sigma_0, sigma_1) for j in range(M)])
        den = np.sum([trial_weight(x_star[j], y_bold, k, j, M, target_dist, delta, delta_1, sigma, sigma_0, sigma_1) for j in range(M)])
        if den != 0:
            alpha = np.minimum(1, num / den)
        else:
            alpha = 1
        r = np.random.rand()

        # algorithm 1, lines 9-12
        if r < alpha:
            X.append(y_bold)
            x = y_bold
        else:
            X.append(x)

plt.plot(np.arange(len(X)), X)
plt.show()