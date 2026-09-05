import pandas as pd
from data_loader.load import *
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats
from scipy.stats import normaltest

train = load_train()
test = load_test_reduced()

tau_array = train.tau[train.has_break] # np.ndarray
m = train.has_break
u = (train.tau[m] + 0.5) / train.online_len[m]

print('break rate  %.4f  p=%.3f' % (m.mean(), stats.binomtest(int(m.sum()), len(train), 0.5).pvalue))
print('KS vs unif  D=%.4f p=%.3f' % stats.kstest(u, 'uniform'))
print('chi2 10-bin p=%.3f' % stats.chisquare(np.histogram(u, bins=10, range=(0,1))[0]).pvalue)
print('mean %.4f (expect 0.5, se %.4f)' % (u.mean(), np.sqrt(1/12/len(u))))
plt.hist(u)
plt.show()
# plt.hist(train.tau[train.has_break] / train.online_len[train.has_break])
# plt.show()


