


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl
from src.fuzzy_rules.fuzzy_rules import read_dataset
from src.util.utils import evaluate_from_hyperparams

file_path = './2020877-serie.txt'
data = read_dataset(file_path)


hyperparams = h = {
  'window_size': 6,
  'steps_forward': 1,
  'nb_sets': 4,
  'aggregation_opt': {'and_func':np.fmin,'or_func': np.fmax} 
}

hyperparams = h = {
  'window_size': 6,
  'steps_forward': 1,
  'nb_sets': 4,
  'aggregation_opt': {'and_func':np.fmin,'or_func': np.fmax} 
}

res = evaluate_from_hyperparams(hyperparams, data, data, True)

print('-'*21 + '\nDataset Distribution\n' + '-'*21)
print('X\t:{} | y_true\t:{}'.format(res['datasets']['X'].shape, res['datasets']['y_true'].shape))
print('X_train\t:{} | y_train\t:{}'.format(res['datasets']['X_train'].shape, res['datasets']['y_train'].shape))
print('X_test\t:{}  | y_test\t:{}'.format(res['datasets']['X_test'].shape, res['datasets']['y_test'].shape))

print('\nmean_squared_error: {}'.format(res['out'][0]))

plt.figure(figsize=(15,5))
plt.plot(res['datasets']['y_true'], '*')
plt.plot(res['out'][1], 's')
plt.show()