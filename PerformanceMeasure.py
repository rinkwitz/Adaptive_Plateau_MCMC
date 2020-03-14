import numpy as np


def lagged_autocovariance(v, t):
    n = len(v)
    return 1 / n * np.sum((v[:n - t] - np.mean(v)) * (v[t:] - np.mean(v)))


def Gamma(v, m):
    return lagged_autocovariance(v, 2 * m) + lagged_autocovariance(v, 2 * m + 1)


def ACT(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    act = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):

            # # initial positive sequence estimator:
            # N = len(simulations[i, :, k])
            # v = simulations[i, :, k].reshape(-1)
            # act[i, k] = -lagged_autocovariance(v, 0)
            # for m in range(N):
            #     if Gamma(v, m) <= 0:
            #         break
            #     act[i, k] += 2 * Gamma(v, m)

            # initial monotone sequence estimator:
            old = np.inf
            N = len(simulations[i, :, k])
            v = simulations[i, :, k].reshape(-1)
            act[i, k] = - lagged_autocovariance(v, 0)
            for m in range(N):
                if Gamma(v, m) <= 0 or Gamma(v, m) > old:
                    break
                act[i, k] += 2 * Gamma(v, m)
                old = Gamma(v, m)

    return act


def log_ACT(simulations):
    return np.log(ACT(simulations))


def ASJD(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    asjd = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):
            asjd[i, k] = np.mean((simulations[i, 1:, k] - simulations[i, :-1, k]) ** 2)
    return asjd
