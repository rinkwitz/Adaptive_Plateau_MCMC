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
            # N = len(simulations[i, :, k])
            # vec = simulations[i, :, k].reshape((-1, 1))
            # cov = 1 / (N - 1) * np.dot(vec - np.mean(vec), vec.T - np.mean(vec))
            # var = 1 / (N - 1) * np.sum((vec - np.mean(vec)) ** 2)
            # act[i, k] = 1 + 2 * np.sum(cov[0, 1:]) / var

            N = len(simulations[i, :, k])
            v = simulations[i, :, k].reshape(-1)
            act[i, k] = - lagged_autocovariance(v, 0)
            for m in range(N):
                if Gamma(v, m) <= 0:
                    break
                act[i, k] += 2 * Gamma(v, m)

    print('median act component-wise:', np.median(act, axis=0))
    return np.log(act)


def log_ACT(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    act = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):
            N = len(simulations[i, :, k])
            v = simulations[i, :, k].reshape(-1)
            act[i, k] = - lagged_autocovariance(v, 0)
            for m in range(N):
                if Gamma(v, m) <= 0:
                    break
                act[i, k] += 2 * Gamma(v, m)
    log_act = np.log(act)
    print('median act component-wise:', np.median(log_act, axis=0))
    return log_act


def ASJD(simulations):
    num_simulations = simulations.shape[0]
    num_components = simulations.shape[2]
    asjd = np.empty((num_simulations, num_components))
    for i in range(num_simulations):
        for k in range(num_components):
            asjd[i, k] = np.mean((simulations[i, 1:, k] - simulations[i, :-1, k]) ** 2)
    print('median asjd component-wise:', np.median(asjd, axis=0))
    return asjd
