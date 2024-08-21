from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

print('\n########   Yuen-Welch Test of mean jump distance  ########')
trim = 0.2
sample1_mean = 0.10
sample1_std = 0.01
sample2_mean = 0.15
sample2_std = 0.01
sample1 = np.random.normal(loc=sample1_mean, scale=sample1_std, size=1000)
sample2 = np.random.normal(loc=sample2_mean, scale=sample2_std, size=500)

result = stats.ttest_ind(sample1, sample2, equal_var=False, trim=0.2)
print(result)
plt.figure(figsize=(8, 6))
plt.title(f'sample1 mean:{sample1_mean}, std:{sample1_std}\nsample2 mean:{sample2_mean}, std:{sample2_std}\np_value:{result.pvalue}')
plt.hist(sample1, alpha=0.5, label='sample1')
plt.hist(sample2, alpha=0.5, label='sample2')
plt.legend()
plt.xlim(0, 0.2)
plt.show()