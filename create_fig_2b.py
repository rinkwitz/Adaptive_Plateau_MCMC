import pathlib
import numpy as np
import matplotlib.pyplot as plt
from Utils import *

path = pathlib.Path('figs')/'fig_2b.png'
M = 5
xs = np.linspace(-10, 10, 10000)
for j in range(M):
    ys = [trial(0.0, t, j, M, 1, 1, .05, .5, .5) for t in xs]
    plt.plot(xs, ys)
plt.legend(('j = 1', 'j = 2', 'j = 3', 'j = 4', 'j = 5'), loc='upper right')
plt.savefig(path)
plt.show()

C = np.sqrt(2 * np.pi * .05 ** 2) / 2 + np.sqrt(2 * np.pi * .5 ** 2) / 2 + 2 * 1
print(1 / C * np.exp(-(10-9)**2/(2*.5**2)) / 2)
