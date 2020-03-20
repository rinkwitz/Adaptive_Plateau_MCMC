import numpy as np
from PerformanceMeasure import *
from Utils import *

show_plots = False
print_latex_tables = True
np.set_printoptions(linewidth=160)

median_header = 'median '
mean_header = 'mean '
min_header = 'min '
max_header = 'max '
median_act = median_log_act = median_asjd = mean_act = mean_log_act = mean_asjd = max_act = max_log_act = max_asjd = min_act = min_log_act = min_asjd = ''
for target_dist in ['pi_1', 'pi_2']:
    path = pathlib.Path('simulations/simulation_{}.npy'.format(target_dist))
    simulations = np.load(path)
    act = ACT(simulations)
    log_act = log_ACT(simulations)
    asjd = ASJD(simulations)
    visualize_ACT(act, target_dist, show_plots)
    visualize_log_ACT(log_act, target_dist, show_plots)
    visualize_ASJD(asjd, target_dist, show_plots)

    median_act += '& {} '.format(str([round(x, 3) for x in np.median(act, axis=0)]).replace('[', '').replace(']', ''))
    median_log_act += '& {} '.format(str([round(x, 3) for x in np.median(log_act, axis=0)]).replace('[', '').replace(']', ''))
    median_asjd += '& {} '.format(str([round(x, 3) for x in np.median(asjd, axis=0)]).replace('[', '').replace(']', ''))

    mean_act += '& {} '.format(str([round(x, 3) for x in np.mean(act, axis=0)]).replace('[', '').replace(']', ''))
    mean_log_act += '& {} '.format(str([round(x, 3) for x in np.mean(log_act, axis=0)]).replace('[', '').replace(']', ''))
    mean_asjd += '& {} '.format(str([round(x, 3) for x in np.mean(asjd, axis=0)]).replace('[', '').replace(']', ''))

    min_act += '& {} '.format(str([round(x, 3) for x in np.min(act, axis=0)]).replace('[', '').replace(']', ''))
    min_log_act += '& {} '.format(str([round(x, 3) for x in np.min(log_act, axis=0)]).replace('[', '').replace(']', ''))
    min_asjd += '& {} '.format(str([round(x, 3) for x in np.min(asjd, axis=0)]).replace('[', '').replace(']', ''))

    max_act += '& {} '.format(str([round(x, 3) for x in np.max(act, axis=0)]).replace('[', '').replace(']', ''))
    max_log_act += '& {} '.format(str([round(x, 3) for x in np.max(log_act, axis=0)]).replace('[', '').replace(']', ''))
    max_asjd += '& {} '.format(str([round(x, 3) for x in np.max(asjd, axis=0)]).replace('[', '').replace(']', ''))

act_str = '\\hline & $\\pi_1$ & $\\pi_2$ \\\\ \\hline\n' + median_header + median_act + '\\\\\\hline \n' + mean_header + mean_act + '\\\\\\hline \n' + min_header + min_act + '\\\\\\hline \n' + max_header + max_act + '\\\\\\hline \n'
log_act_str = '\\hline & $\\pi_1$ & $\\pi_2$ \\\\ \\hline\n' + median_header + median_log_act + '\\\\\\hline \n' + mean_header + mean_log_act + '\\\\\\hline \n' + min_header + min_log_act + '\\\\\\hline \n' + max_header + max_log_act + '\\\\\\hline \n'
asjd_str = '\\hline & $\\pi_1$ & $\\pi_2$ \\\\ \\hline\n' + median_header + median_asjd + '\\\\\\hline \n' + mean_header + mean_asjd + '\\\\\\hline \n' + min_header + min_asjd + '\\\\\\hline \n' + max_header + max_asjd + '\\\\\\hline \n'

median_act = median_log_act = median_asjd = mean_act = mean_log_act = mean_asjd = max_act = max_log_act = max_asjd = min_act = min_log_act = min_asjd = ''
for target_dist in ['pi_3', 'pi_4']:
    path = pathlib.Path('simulations/simulation_{}.npy'.format(target_dist))
    simulations = np.load(path)
    act = ACT(simulations)
    log_act = log_ACT(simulations)
    asjd = ASJD(simulations)
    visualize_ACT(act, target_dist, show_plots)
    visualize_log_ACT(log_act, target_dist, show_plots)
    visualize_ASJD(asjd, target_dist, show_plots)

    median_act += '& {} '.format(str([round(x, 3) for x in np.median(act, axis=0)]).replace('[', '').replace(']', ''))
    median_log_act += '& {} '.format(str([round(x, 3) for x in np.median(log_act, axis=0)]).replace('[', '').replace(']', ''))
    median_asjd += '& {} '.format(str([round(x, 3) for x in np.median(asjd, axis=0)]).replace('[', '').replace(']', ''))

    mean_act += '& {} '.format(str([round(x, 3) for x in np.mean(act, axis=0)]).replace('[', '').replace(']', ''))
    mean_log_act += '& {} '.format(str([round(x, 3) for x in np.mean(log_act, axis=0)]).replace('[', '').replace(']', ''))
    mean_asjd += '& {} '.format(str([round(x, 3) for x in np.mean(asjd, axis=0)]).replace('[', '').replace(']', ''))

    min_act += '& {} '.format(str([round(x, 3) for x in np.min(act, axis=0)]).replace('[', '').replace(']', ''))
    min_log_act += '& {} '.format(str([round(x, 3) for x in np.min(log_act, axis=0)]).replace('[', '').replace(']', ''))
    min_asjd += '& {} '.format(str([round(x, 3) for x in np.min(asjd, axis=0)]).replace('[', '').replace(']', ''))

    max_act += '& {} '.format(str([round(x, 3) for x in np.max(act, axis=0)]).replace('[', '').replace(']', ''))
    max_log_act += '& {} '.format(str([round(x, 3) for x in np.max(log_act, axis=0)]).replace('[', '').replace(']', ''))
    max_asjd += '& {} '.format(str([round(x, 3) for x in np.max(asjd, axis=0)]).replace('[', '').replace(']', ''))

act_str += '\\hline & $\\pi_3$ & $\\pi_4$ \\\\ \\hline\n' + median_header + median_act + '\\\\\\hline \n' + mean_header + mean_act + '\\\\\\hline \n' + min_header + min_act + '\\\\\\hline \n' + max_header + max_act + '\\\\\\hline \n'
log_act_str += '\\hline & $\\pi_3$ & $\\pi_4$ \\\\ \\hline\n' + median_header + median_log_act + '\\\\\\hline \n' + mean_header + mean_log_act + '\\\\\\hline \n' + min_header + min_log_act + '\\\\\\hline \n' + max_header + max_log_act + '\\\\\\hline \n'
asjd_str += '\\hline & $\\pi_3$ & $\\pi_4$ \\\\ \\hline\n' + median_header + median_asjd + '\\\\\\hline \n' + mean_header + mean_asjd + '\\\\\\hline \n' + min_header + min_asjd + '\\\\\\hline \n' + max_header + max_asjd + '\\\\\\hline \n'

if print_latex_tables:
    print('act:\n{}\n'.format(act_str))
    print('log_act:\n{}\n'.format(log_act_str))
    print('asjd:\n{}\n'.format(asjd_str))