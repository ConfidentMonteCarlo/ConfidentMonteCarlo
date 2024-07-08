# Note: This file must be in the same directory as cmc.py and password_guessing_model.py to work properly
#       It is put in a seperate 'examples' directory for demonstration purposes.
#       Also, make sure to untar the dataset files as github disallows files larger than 100 MB.

import math
import matplotlib.pyplot as plt
import password_guessing_model as pgm
from cmc import ConfidentMonteCarlo, MeshPointGenerator

mod1 = pgm.PCFGModel('./dataset/testset.txt')
cmc1 = ConfidentMonteCarlo(mod1, './dataset/testset.csv')
mod2 = pgm.NGramModel('./dataset/testset.txt', 4)
cmc2 = ConfidentMonteCarlo(mod2, './dataset/testset.csv')

mpg = MeshPointGenerator()

def pol(pwd): # length > 6 and contains digit and uppercase letter
    return len(s) >= 6 and any(c.isdigit() for c in s) and any(c.isupper() for c in s)
mod3 = pgm.PCFGModel('./dataset/testset.txt')
cmc3 = ConfidentMonteCarlo(mod3, './dataset/testset.csv', pol)

samples1 = cmc1.sample(5000000)
cmc1.group_sample(2500)
samples2 = cmc2.sample(5000000)
cmc2.group_sample(2500)

print(cmc1.hoeffding_bound(math.log2(1e-7), 0.01))
print(cmc2.markov_lowerbound(math.log2(1e-13), 0.01))

mesh = mpg.from_sample(samples1, 100, math.log2(1e-14))

cmc1.dataset_curve_bound_fit(mesh, 0.01) # note: user can also fit seperately with different mesh points and error rates
print(cmc1.dataset_curve_bound1_query(1e8))
dataset_curves_1 = cmc1.dataset_curve_bound_plot(savename='./plots/pcfg_dataset_curve_bound.png', title='pcfg (5M samples)')

cmc1.population_curve_bound1_fit(mesh, 0.005, 0.005)
cmc1.population_curve_bound2_fit(mesh, 0.005, 0.005)
cmc1.population_curve_bound3_fit(0.01)
print(cmc1.population_curve_bound2_query(1e10))
population_curves_1 = cmc1.population_curve_bound_plot(savename='./plots/pcfg_population_curve_bound.png', title='pcfg (5M samples)')

mesh = mpg.from_sample(samples2, 100, math.log2(1e-14))
cmc2.dataset_curve_bound_fit(mesh, 0.01)
dataset_curves_2 = cmc2.dataset_curve_bound_plot()

fig, ax = plt.subplots()
ax.stairs(dataset_curves_1['lb1'][0], dataset_curves_1['lb1'][1], linewidth=0.8, baseline=None, label='pcfg lb1', color='red', linestyle='solid')
ax.stairs(dataset_curves_1['ub1'][0], dataset_curves_1['ub1'][1], linewidth=0.8, baseline=None, label='pcfg ub1', color='red', linestyle='dashed')
ax.stairs(dataset_curves_2['lb1'][0], dataset_curves_2['lb1'][1], linewidth=0.8, baseline=None, label='4gram lb1', color='blue', linestyle='solid')
ax.stairs(dataset_curves_2['ub1'][0], dataset_curves_2['ub1'][1], linewidth=0.8, baseline=None, label='4gram ub1', color='blue', linestyle='dashed')
ax.set_xlabel('log10(guessing number)')
ax.set_ylabel('fraction of cracked passwords')
ax.set_title('pcfg vs. 4gram (5M samples)')
ax.legend()
fig.set_size_inches(12, 7)
fig.savefig('./plots/pcfg_v_4gram.png', dpi=300)

