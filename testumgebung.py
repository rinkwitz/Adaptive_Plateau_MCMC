import numpy as np
from PerformanceMeasure import *
from Utils import *

target_dist = 'pi_3'
simulations = np.load('simulations/simulation_{}.npy'.format(target_dist))
act = ACT(simulations)
asjd = ASJD(simulations)
visualize_ACT(act, target_dist)
visualize_ASJD(asjd, target_dist)
