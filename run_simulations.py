import time
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from PerformanceMeasure import *
from multiprocessing import Pool


def run(args):
    x_0, r = args
    X = []
    x = x_0.copy()
    X.append(x_0.copy())

    # algorithm 1, line 2
    for n in range(N):

        # optional:
        if (n + 1) % show_status_interval == 0 and show_status:
            print('r:', r + 1, '\tn:', n + 1, '\tx:', x,
                  '\t{} % done / estimated time left {}s ...'.format(round(100 * (r * N + n + 1) / (N * R), 2), int(
                      (time.time() - start) / ((r * N + n + 1) / (N * R)) - time.time() + start)))

        # algorithm 2, lines 1-3
        if n == 0:
            c_1 = {k: 0 for k in range(len(x))}
            c_M = {k: 0 for k in range(len(x))}

        # algorithm 2, lines 4-3
        if (n + 1) % L == 0:
            if use_adaption and (not use_burn_in or n < burn_in * N):
                # slightly different than in paper ...
                if show_adaption_info:
                    print('frequencies of center proposals:', [c_1[k] / L for k in range(len(c_1))])
                    print('frequencies of tail proposals:', [c_M[k] / L for k in range(len(c_M))])
                P_n = np.maximum(.99 ** n, 1 / np.sqrt(n + 1))
                for k in range(len(x)):
                    if np.random.rand() < P_n:
                        if c_1[k] > eta_1 * L:
                            delta[k] *= .5
                            delta_1[k] *= .5
                        if c_M[k] > eta_2 * L:
                            delta[k] *= 2
                            delta_1[k] *= 2
                if show_adaption_info:
                    print('current delta/delta_1:', delta)
            c_1 = {k: 0 for k in range(len(x))}
            c_M = {k: 0 for k in range(len(x))}

        # algorithm 1, line 3
        for k in range(len(x)):

            # algorithm 1, line 4
            z = [rejection_sampling_trial(x[k], j, M, delta[k], delta_1[k], sigma, sigma_0, sigma_1) for j in range(M)]

            # algorithm 1, line 5
            w = [trial_weight(z[j], x, k, j, M, target_dist, delta[k], delta_1[k], sigma, sigma_0, sigma_1) for j in
                 range(M)]

            # algorithm 1, line 6
            y, index = draw_from_z_proportional_to_w(z, w)
            # count for adaption:
            if index == 0:
                c_1[k] += 1
            elif index == M - 1:
                c_M[k] += 1

            # algorithm 1, line 7
            x_star = [rejection_sampling_trial(y, j, M, delta[k], delta_1[k], sigma, sigma_0, sigma_1) for j in
                      range(M - 1)]
            x_star.append(x[k])

            # algorithm 1, line 8
            y_bold = x.copy()
            y_bold[k] = y
            num = np.sum(
                [trial_weight(z[j], x, k, j, M, target_dist, delta[k], delta_1[k], sigma, sigma_0, sigma_1) for j in
                 range(M)])
            den = np.sum(
                [trial_weight(x_star[j], y_bold, k, j, M, target_dist, delta[k], delta_1[k], sigma, sigma_0, sigma_1)
                 for j in range(M)])
            if den != 0:
                alpha = np.minimum(1, num / den)
            else:
                alpha = 1

            # algorithm 1, lines 9-13
            if np.random.rand() < alpha:
                x = y_bold.copy()

        X.append(x.copy())

    return [sample.copy() for sample in X]


# adjustable parameters:
np.set_printoptions(linewidth=160)
dim_dict = {'pi_test': 1,
            'pi_1': 4,
            'pi_2': 8,
            'pi_3': 2,
            'pi_4': 1}
N_dict = {'pi_test': 1000,
            'pi_1': 4000,
            'pi_2': 10000,
            'pi_3': 3000,
            'pi_4': 3000}
M = 5
sigma = .05
sigma_0 = 3.
sigma_1 = 3.
use_adaption = True
eta_1 = .4
eta_2 = .4
L = 40
R = 200
use_burn_in = True
burn_in = .5
show_status = True
show_status_interval = 500
show_adaption_info = False
target_dists = ['pi_1', 'pi_2', 'pi_3', 'pi_4']

for target_dist in target_dists:
    #adjustable deltas:
    delta = [2. for i in range(dim_dict[target_dist])]
    delta_1 = [2. for i in range(dim_dict[target_dist])]

    N = N_dict[target_dist]
    x_initial = np.random.randn(dim_dict[target_dist])
    start = time.time()

    p = Pool(4)
    simulations = p.map(run, [(x_initial.copy(), r) for r in range(R)])
    simulations = np.array(simulations)
    np.save(pathlib.Path('simulations')/'simulation_{}.npy'.format(target_dist), simulations)
