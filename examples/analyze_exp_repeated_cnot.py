import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion

corrected_exp = np.load('corrected_exp.npy')
log_probabilities_exp = np.load('log_probabilities_exp.npy')
observable_flips_exp = np.load('observable_flips_exp.npy')

predictions_theory_3 = np.load('predictions_theory_3.npy')
log_probabilities_theory_3 = np.load('log_probabilities_theory_3.npy')
observable_flips_theory_3 = np.load('observable_flips_theory_3.npy')
corrected_theory_3 = np.load('corrected_theory_3.npy')



predictions_theory = np.load('predictions_theory.npy')
log_probabilities_theory = np.load('log_probabilities_theory.npy')
observable_flips_theory = np.load('observable_flips_theory.npy')
corrected_theory = np.load('corrected_theory.npy')

logical_gaps_theory = np.diff(log_probabilities_theory, axis=1).flatten()
acceptance_fractions = np.linspace(.05, 1, 20)
order = np.argsort(logical_gaps_theory.flatten())
# Sort by logical gap
corrected_theory = corrected_theory[order].flatten()

logical_gaps_exp = np.diff(log_probabilities_exp, axis=1).flatten()
order = np.argsort(logical_gaps_exp.flatten())
# Sort by logical gap
corrected_exp = corrected_exp[order].flatten()


logical_gaps_theory_3 = np.diff(log_probabilities_theory_3, axis=1).flatten()
order = np.argsort(logical_gaps_theory_3.flatten())
# Sort by logical gap
corrected_theory_3 = corrected_theory_3[order].flatten()


plt.plot(acceptance_fractions, [1-np.mean(corrected_theory[:int(len(corrected_theory)*acceptance_fraction)]) for acceptance_fraction in acceptance_fractions], marker='o', color='blue', markeredgecolor='navy', label='theory d=5')
plt.plot(acceptance_fractions, [1-np.mean(corrected_theory_3[:int(len(corrected_theory_3)*acceptance_fraction)]) for acceptance_fraction in acceptance_fractions], marker='o', color='lightblue', markeredgecolor='blue', label='theory d=3')
plt.plot(acceptance_fractions, [1-np.mean(corrected_exp[:int(len(corrected_exp)*acceptance_fraction)]) for acceptance_fraction in acceptance_fractions], marker='o', color='red', markeredgecolor='firebrick', label='exp')
plt.xlabel('Acceptance fraction')
plt.ylabel('Probability of logical $+1$')
plt.legend()
plt.show()




"""

num_bins = 7
binned = np.zeros(num_bins)
binned_total = np.zeros(num_bins)
binned_correct = np.zeros(num_bins)
bin_size = len(corrected_exp) // num_bins
print(bin_size)

for i in range(num_bins):
    binned[i] = np.mean(corrected_exp[bin_size*i:bin_size*(i+1)])
    binned_correct[i] = np.sum(corrected_exp[bin_size * i:bin_size * (i + 1)])
    binned_total[i] = len(corrected_exp[bin_size * i:bin_size * (i + 1)])
lower, upper = statsmodels.stats.proportion.proportion_confint(binned_correct, binned_total, alpha=0.30, method='beta')
print(lower)
print(upper)
plt.ylabel('logical error rate')
plt.xlabel('time bin')
plt.errorbar(np.arange(len(binned)), binned, yerr=[binned - lower, upper - binned], marker='o')
plt.show()
"""