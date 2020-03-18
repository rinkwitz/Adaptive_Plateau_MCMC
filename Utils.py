import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *


def plateau_dist_exp_decaying_tails_pdf(y, mu, delta, sigma_1, sigma_2):
    C = np.sqrt(2 * np.pi * sigma_1 ** 2) / 2 + np.sqrt(2 * np.pi * sigma_2 ** 2) / 2 + 2 * delta
    if y < mu - delta:
        return np.exp(-(y - (mu - delta)) ** 2 / (2 * sigma_1 ** 2)) / C
    elif mu - delta <= y <= delta + mu:
        return 1 / C
    else:
        return np.exp(-(y - (mu + delta)) ** 2 / (2 * sigma_2 ** 2)) / C


def trial(x, y, j, M, delta, delta_1, sigma, sigma_0, sigma_1):
    # k not specified
    if j == 0:
        return plateau_dist_exp_decaying_tails_pdf(y, x, delta_1, sigma, sigma)
    elif j < M - 1:
        return (plateau_dist_exp_decaying_tails_pdf(y, x - (2 * j - 1) * delta - delta_1, delta, sigma,
                                                    sigma) + plateau_dist_exp_decaying_tails_pdf(y, x + (
                2 * j - 1) * delta + delta_1, delta, sigma, sigma)) / 2
    else:
        return (plateau_dist_exp_decaying_tails_pdf(y, x - (2 * j - 1) * delta - delta_1, delta, sigma_0,
                                                    sigma) + plateau_dist_exp_decaying_tails_pdf(y, x + (
                2 * j - 1) * delta + delta_1, delta, sigma, sigma_1)) / 2


def rejection_sampling_trial(x, j, M, delta, delta_1, sigma, sigma_0, sigma_1):
    tail_lowest = 1e-4
    if j == 0:
        C = np.sqrt(2 * np.pi * sigma ** 2) / 2 + np.sqrt(2 * np.pi * sigma ** 2) / 2 + 2 * delta_1
        tail_width = np.sqrt(np.log(C * tail_lowest) * (-2 * sigma ** 2))
        lower_bound = x - delta_1 - tail_width
        upper_bound = x + delta_1 + tail_width
        g = 1 / (upper_bound - lower_bound)
        c = g / C
    elif j < M - 1:
        C = np.sqrt(2 * np.pi * sigma ** 2) / 2 + np.sqrt(2 * np.pi * sigma ** 2) / 2 + 2 * delta
        tail_width = np.sqrt(np.log(2 * C * tail_lowest) * (-2 * sigma ** 2))
        lower_bound_1 = x - 2 * j * delta - delta_1 - tail_width
        upper_bound_1 = x - 2 * (j - 1) * delta - delta_1 + tail_width
        lower_bound_2 = x + 2 * (j - 1) * delta + delta_1 - tail_width
        upper_bound_2 = x + 2 * j * delta + delta_1 + tail_width
        g = 1 / (upper_bound_1 - lower_bound_1)
        c = g / C
    else:
        C = np.sqrt(2 * np.pi * sigma ** 2) / 2 + np.sqrt(2 * np.pi * sigma_0 ** 2) / 2 + 2 * delta
        tail_width_1 = np.sqrt(np.log(2 * C * tail_lowest) * (-2 * sigma ** 2))
        tail_width_2 = np.sqrt(np.log(2 * C * tail_lowest) * (-2 * sigma_0 ** 2))
        lower_bound_1 = x - 2 * j * delta - delta_1 - tail_width_2
        upper_bound_1 = x - 2 * (j - 1) * delta - delta_1 + tail_width_1
        lower_bound_2 = x + 2 * (j - 1) * delta + delta_1 - tail_width_1
        upper_bound_2 = x + 2 * j * delta + delta_1 + tail_width_2
        g = 1 / (upper_bound_1 - lower_bound_1)
        c = g / C
    while True:
        if j == 0:
            Y = np.random.uniform(lower_bound, upper_bound)
        elif j < M - 1:
            if np.random.rand() < .5:
                Y = np.random.uniform(lower_bound_1, upper_bound_1)
            else:
                Y = np.random.uniform(lower_bound_2, upper_bound_2)
        else:
            if np.random.rand() < .5:
                Y = np.random.uniform(lower_bound_1, upper_bound_1)
            else:
                Y = np.random.uniform(lower_bound_2, upper_bound_2)
        U = np.random.uniform()
        if U < trial(x, Y, j, M, delta, delta_1, sigma, sigma_0, sigma_1) / (c * g):
            return Y


def trial_weight(z, x, k, j, M, target_dist, delta, delta_1, sigma, sigma_0, sigma_1):
    x_replacement = x.copy()
    x_replacement[k] = z

    if target_dist == 'pi_test':
        pi = norm(loc=100, scale=3).pdf(x_replacement)

    if target_dist == 'pi_1':
        # mixture of Gaussians (4-dim)
        mu_1 = np.array([5, 5, 0, 0])
        mu_2 = np.array([15, 15, 0, 0])
        Sigma_1 = np.diag([6.25, 6.25, 6.25, 0.01])
        Sigma_2 = np.diag([6.25, 6.25, .25, 0.01])
        pi = multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(x_replacement) / 2 + multivariate_normal(mean=mu_2,
                                                                                                      cov=Sigma_2).pdf(
            x_replacement) / 2

    if target_dist == 'pi_2':
        # banana distribution (8-dim)
        b = .03
        phi = x_replacement.copy()
        phi[1] = phi[1] + b * phi[0] ** 2 - 100 * b
        Sigma_3 = np.diag([100, 1, 1, 1, 1, 1, 1, 1])
        pi = multivariate_normal(mean=np.zeros(8), cov=Sigma_3).pdf(phi)

    if target_dist == 'pi_3':
        # perturbed 2-dimensional Gaussian
        from NormalizingConstant_pi_3 import pi_3_normalizing_const
        A = np.array([[1, 1], [1, 1.5]])
        x_replacement = x_replacement.reshape((-1, 1))
        pi = pi_3_normalizing_const * (np.exp(
            -np.dot(np.dot(x_replacement.T, A), x_replacement) - np.cos(x_replacement[0, 0] / .1) - .5 * np.cos(
                x_replacement[1, 0] / .1)))[0, 0]

    if target_dist == 'pi_4':
        # 1D bi-stable distribution
        from NormalizingConstant_pi_4 import pi_4_normalizing_const
        pi = pi_4_normalizing_const * np.exp(-x_replacement ** 4 + 5 * x_replacement ** 2 - np.cos(x_replacement / .02))

    return pi * trial(x[k], z, j, M, delta, delta_1, sigma, sigma_0, sigma_1) * lambda_function(x[k], z, j, M, delta, delta_1, sigma, sigma_0, sigma_1)


def lambda_function(x, y, j, M, delta, delta_1, sigma, sigma_0, sigma_1):
    alpha = 2.5
    return np.abs(x - y) ** alpha
    #return trial(x, y, j, M, delta, delta_1, sigma, sigma_0, sigma_1) * np.abs(x - y) ** alpha


def draw_from_z_proportional_to_w(z, w):
    if np.sum(w) != 0:
        ps = np.array(w) / np.sum(w)
        u = np.random.rand()
        p_cumulative = 0.0
        for i, p in enumerate(ps):
            p_cumulative += p
            if u <= p_cumulative:
                return z[i], i
    else:
        return z[0], 0


def visualize_ACT(act, target_dist, show_plot = True):
    num_components = act.shape[1]
    fig, axes = plt.subplots(1, num_components)
    fig.set_size_inches(9.6, 4.8)
    for k in range(num_components):
        if num_components != 1:
            ax = axes[k]
        else:
            ax = axes
        data = act[:, k]
        ax.set_xlabel('component {}'.format(k + 1))
        ax.set_ylabel('ACT')
        ax.set_xticks([])
        ax.violinplot(data, showmeans=False, showmedians=True)
    plt.tight_layout()
    plt.savefig(pathlib.Path('figs')/'ACT_{}.png'.format(target_dist))
    if show_plot:
        plt.show()
    plt.close()

def visualize_log_ACT(log_act, target_dist, show_plot = True):
    num_components = log_act.shape[1]
    fig, axes = plt.subplots(1, num_components)
    fig.set_size_inches(9.6, 4.8)
    for k in range(num_components):
        if num_components != 1:
            ax = axes[k]
        else:
            ax = axes
        data = log_act[:, k]
        ax.set_xlabel('component {}'.format(k + 1))
        ax.set_ylabel('log_ACT')
        ax.set_xticks([])
        ax.violinplot(data, showmeans=False, showmedians=True)
    plt.tight_layout()
    plt.savefig(pathlib.Path('figs')/'log_ACT_{}.png'.format(target_dist))
    if show_plot:
        plt.show()
    plt.close()


def visualize_ASJD(asjd, target_dist, show_plot = True):
    num_components = asjd.shape[1]
    fig, axes = plt.subplots(1, num_components)
    fig.set_size_inches(9.6, 4.8)
    for k in range(num_components):
        if num_components != 1:
            ax = axes[k]
        else:
            ax = axes
        data = asjd[:, k]
        ax.set_xlabel('component {}'.format(k + 1))
        ax.set_ylabel('ASJD')
        ax.set_xticks([])
        ax.violinplot(data, showmeans=False, showmedians=True)
    plt.tight_layout()
    plt.savefig(pathlib.Path('figs')/'ASDJ_{}.png'.format(target_dist))
    if show_plot:
        plt.show()
    plt.close()
