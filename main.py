import numpy as np
from Utils import *

M = 5
N = 3000
x = np.random.randn(4)
X = [x]
for n in range(N):
    print(n, x)
    for k in range(len(x)):
        # 4
        z = [rejection_sampling_trial(x[k], j, M) for j in range(M)]

        # 5
        w = [trial_weight(z[j], x, k, j, M) for j in range(M)]

        # 6
        y = draw_from_z_proportional_to_w(z, w)

        # 7
        x_star = [rejection_sampling_trial(y, j, M) for j in range(M - 1)]
        x_star.append(x[k])

        # 8
        y_bold = x.copy()
        y_bold[k] = y
        num = np.sum([trial_weight(z[j], x, k, j, M) for j in range(M)])
        den = np.sum([trial_weight(x_star[j], y_bold, k, j, M) for j in range(M)])
        if den != 0:
            alpha = np.minimum(1, num / den)
        else:
            alpha = 1
        r = np.random.rand()

        # 9-12
        if r < alpha:
            X.append(y_bold)
            x = y_bold
        else:
            X.append(x)


        #print(w)
        # for j in range(M):
        #     print(j)
        #     xs = np.linspace(x[k] - 30, x[k] + 30, 200)
        #     ys = [trial_proposal(i, x[k], j, M) for i in xs]
        #     z = [rejection_sampling_trial(x[k], j, M) for j2 in range(20)]
        #     plt.plot(xs, ys)
        #     plt.scatter(z, np.zeros(len(z)))
        #     #break
        # plt.show()
        # break

print(X[0])
print(X[-1])