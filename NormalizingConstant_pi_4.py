import numpy as np
from scipy.integrate import quad

pi_4_normalizing_const = 1 / quad(lambda x: np.exp(-x ** 4 + 5 * x ** 2 - np.cos(x / .02)), -np.inf, np.inf)[
    0]
