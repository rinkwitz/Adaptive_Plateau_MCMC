import pathlib
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from scipy.integrate import dblquad


def integrand(x_1, x_2, x_3, x_4):
    vec = np.array([x_1, x_2, x_3, x_4])
    return multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(vec) / 2 + multivariate_normal(mean=mu_2, cov=Sigma_2).pdf(vec) / 2

path = pathlib.Path('figs')/'fig_3.png'
fig, ax = plt.subplots(2, 2)

# plot (a)
a = 1
res = a * 7 - a + 1
x = np.linspace(-5, 25, res)
y = np.linspace(-5, 25, res)
X, Y = np.meshgrid(x, y)
mu_1 = np.array([5, 5, 0, 0])
mu_2 = np.array([15, 15, 0, 0])
Sigma_1 = np.diag([6.25, 6.25, 6.25, 0.01])
Sigma_2 = np.diag([6.25, 6.25, .25, 0.01])

lower = -5
upper = 25
step = 1
tensor = [np.array([lower+i, lower+j, lower+k, lower+l]) for i in range(0,upper-lower,step)
          for j in range(0,upper-lower,step) for k in range(0,upper-lower,step) for l in range(0,upper-lower,step)]
print(tensor)
#
Z = multivariate_normal(mean=mu_1, cov=Sigma_1).pdf(tensor) / 2 + multivariate_normal(mean=mu_2, cov=Sigma_2).pdf(tensor) / 2
print(Z.shape)
print(Z)
# im = ax[0, 0].imshow(Z, cmap='Greys')
# ticks = [max(a * i, 0) for i in range(7)]
# labels = [int(x[t]) for t in ticks]
# ax[0, 0].set_xticks(ticks)
# ax[0, 0].set_xticklabels(labels)
# fig.colorbar(im, ax=ax[0, 0])
#
# plt.tight_layout()
# plt.savefig(path)
# plt.show()
