import pathlib
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from scipy.integrate import dblquad


def integrand(x_1, x_2, x_3, x_4):
    vec = np.array([x_1, x_2, x_3, x_4])
    return multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(vec) / 2 + multivariate_normal(mean=mu_2, cov=Sigma_2).pdf(vec) / 2

path = pathlib.Path('figs')/'fig_3.png'
fig, ax = plt.subplots()
x = np.linspace(-5, 25, 100)
y = np.linspace(-5, 25, 100)
X, Y = np.meshgrid(x, y)
mu_1 = np.array([5, 5, 0, 0])
mu_2 = np.array([15, 15, 0, 0])
Sigma_1 = np.diag([6.25, 6.25, 6.25, 0.01])
Sigma_2 = np.diag([6.25, 6.25, .25, 0.01])
tensor = np.zeros((X.shape[0], X.shape[1], 4))
tensor[:, :, 0] = X
tensor[:, :, 1] = Y
Z = multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(tensor) / 2 + multivariate_normal(mean=mu_2, cov=Sigma_2).pdf(tensor) / 2
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         print(i, j)
#         #vec = np.array([X[i, j], Y[i, j], 0, 0])
#         Z[i, j] = dblquad(lambda x_4, x_3: integrand(X[i, j], Y[i, j], x_3, x_4), -np.inf, np.inf, lambda x_3: -np.inf, lambda x_3: np.inf)[0]

im = ax.imshow(Z, cmap='Greys')
#ticks = [20 * i for i in range(5)] + [99]
#axes[0, 0].set_xticks(ticks)
#axes[0, 0].set_xticklabels([x[t] for t in ticks])
fig.colorbar(im, ax=ax)
#axes[0, 0].set_title('a')
plt.savefig(path)

plt.show()
