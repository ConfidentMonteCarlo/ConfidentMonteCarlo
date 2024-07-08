# Note: This file must be in the same directory as cmc.py and password_guessing_model.py to work properly
#       It is put in a seperate 'examples' directory for demonstration purposes.
#       Also, make sure to untar the dataset files as github disallows files larger than 100 MB.

import math
import password_guessing_model as pgm
from cmc import ConfidentMonteCarlo

mod = pgm.PCFGModel('./dataset/testset.txt')
cmc = ConfidentMonteCarlo(mod, './dataset/testset.csv')

print(cmc.guessing_number_bound('Password123', 0.01))
print(cmc.guessing_number_bound(math.log2(1e-15), 0.01))
cmc.guessing_number_plot(0.01, 'PCFG Model: log probability vs. guessing number', './plots/pcfg_guessing_number_plot.png')

print(cmc.dataset_guessing_curve_bound(1e7, 0.01))
cmc.population_guessing_curve_plot(0.01, 'PCFG Model: Population Guessing Curve', './plots/pcfg_guessing_curve.png')
