from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

print('\n########   Chi-square test  ########')
tp1 = 1
fp1 = 1117
tp2 = 1
fp2 = 146
table = np.array([[tp1, tp2], [fp1, fp2]])
print('table:\n', table)
res = stats.chi2_contingency(table)
print(f'statstic:{res[0]}, p-value:{res[1]}')
