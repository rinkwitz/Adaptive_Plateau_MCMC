import numpy as np
from PerformanceMeasure import *
from Utils import *

show_plots = False
np.set_printoptions(linewidth=160)

for target_dist in ['pi_3', 'pi_4']:
#for target_dist in ['pi_1', 'pi_2', 'pi_3', 'pi_4']:
    path = pathlib.Path('simulations/simulation_{}.npy'.format(target_dist))
    if path.exists():
        simulations = np.load(path)
        act = ACT(simulations)
        log_act = log_ACT(simulations)
        asjd = ASJD(simulations)
        visualize_ACT(act, target_dist, show_plots)
        visualize_log_ACT(log_act, target_dist, show_plots)
        visualize_ASJD(asjd, target_dist, show_plots)
        print('target distribution: {}\tmedian act component-wise: {}'.format(target_dist, np.median(act, axis=0)))
        print('target distribution: {}\tmedian log_act component-wise: {}'.format(target_dist, np.median(log_act, axis=0)))
        print('target distribution: {}\tmedian asjd component-wise: {}\n'.format(target_dist, np.median(asjd, axis=0)))