import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from scipy.integrate import dblquad
from scipy import integrate

show_plots = False

# (a)
path = pathlib.Path('figs')/'fig_3_pi_1.png'
fig, ax = plt.subplots(1, 1)
res = 49

x = np.linspace(-5, 25, res)
y = np.linspace(-5, 25, res)
X, Y = np.meshgrid(x, y)

def f(x_1, x_2, x_3, x_4):
    v = np.array([x_1, x_2, x_3, x_4])
    return multivariate_normal.pdf(v, mean=mu_1, cov=Sigma_1) / 2 + multivariate_normal.pdf(v, mean=mu_2, cov=Sigma_2) / 2

mu_1 = np.array([5, 5, 0, 0])
mu_2 = np.array([15, 15, 0, 0])
Sigma_1 = np.diag([6.25, 6.25, 6.25, 0.01])
Sigma_2 = np.diag([6.25, 6.25, .25, 0.01])
Z = np.empty((res, res))
start = time.time()
for i in range(res):
    for j in range(res):
        x_1 = X[i, j]
        x_2 = Y[i, j]
        Z[i, j] = integrate.nquad(lambda x_3, x_4: f(x_1, x_2, x_3, x_4), [[-np.inf, np.inf], [-np.inf, np.inf]], opts=[{'epsabs': .01} for l in range(2)])[0]
        print(i, j, 'time left:', (time.time() - start) / ((i * res + j + 1) / res ** 2) - time.time() + start)
ticks = [(res // 6) * i for i in range(7)]
labels = [int(round(x[t], 0)) for t in ticks]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

im = ax.imshow(Z, cmap='Greys')
fig.colorbar(im, ax=ax)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(path)
if show_plots:
    plt.show()
plt.close()


# (b)
path = pathlib.Path('figs')/'fig_3_pi_2.png'
fig, ax = plt.subplots(1, 1)
res = 49

x = np.linspace(-30, 30, res)
y = np.linspace(-15, 15, res)
X, Y = np.meshgrid(x, y)
mu_3 = np.zeros(4)
Sigma_3 = np.diag([100, 1, 1, 1])

def g(x_1, x_2, x_3, x_4):
    v = np.array([x_1, x_2 + .03 * x_1 ** 2 - 3, x_3, x_4])
    return multivariate_normal.pdf(v, mean=mu_3, cov=Sigma_3)

Z = np.empty((res, res))
start = time.time()
for i in range(res):
    for j in range(res):
        x_1 = X[i, j]
        x_2 = Y[i, j]
        Z[i, j] = integrate.nquad(lambda x_3, x_4: g(x_1, x_2, x_3, x_4), [[-np.inf, np.inf], [-np.inf, np.inf]], opts=[{'epsabs': .001} for l in range(2)])[0]
        print(i, j, Z[i, j], 'time left:', (time.time() - start) / ((i * res + j + 1) / res ** 2) - time.time() + start)
ticks = [(res // 6) * i for i in range(7)]
xlabels = [int(round(x[t], 0)) for t in ticks]
ylabels = [int(round(y[t], 0)) for t in ticks]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

im = ax.imshow(Z, cmap='Greys')
fig.colorbar(im, ax=ax)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(path)
if show_plots:
    plt.show()
plt.close()


# (c)
path = pathlib.Path('figs')/'fig_3_pi_3.png'
fig, ax = plt.subplots(1, 1)
res = 500

x = np.linspace(-3, 3, res)
y = np.linspace(-3, 3, res)
X, Y = np.meshgrid(x, y)

from NormalizingConstant_pi_3 import pi_3_normalizing_const
Z = np.empty((res, res))
A = np.array([[1, 1], [1, 1.5]])
for i in range(res):
    for j in range(res):
        v = np.array([[X[i, j]], [Y[i, j]]])
        Z[i, j] = pi_3_normalizing_const * (np.exp(-np.dot(np.dot(v.T, A), v) - np.cos(v[0, 0] / .1) - .5 * np.cos(v[1, 0] / .1)))[0, 0]
ticks = [(res // 6) * i for i in range(7)]
labels = [int(round(x[t], 0)) for t in ticks]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

im = ax.imshow(Z, cmap='Greys')
fig.colorbar(im, ax=ax)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(path)
if show_plots:
    plt.show()
plt.close()


# (d)
path = pathlib.Path('figs')/'fig_3_pi_4.png'
fig, ax = plt.subplots(1, 1)
res = 1000

from NormalizingConstant_pi_4 import pi_4_normalizing_const
xs = np.linspace(-3, 3, res)
ys = pi_4_normalizing_const * np.exp(-xs ** 4 + 5 * xs ** 2 - np.cos(xs / .02))
ax.plot(xs, ys, 'k')

plt.tight_layout()
plt.savefig(path)
if show_plots:
    plt.show()
plt.close()
