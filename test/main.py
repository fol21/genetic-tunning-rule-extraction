import random
import numpy as np
# import matplotlib.pyplot as plt

from skfuzzy import control as ctrl
from src.rule_extraction.fuzzy_rules import read_dataset
from src.rule_extraction.utils import evaluate_from_hyperparams

from sklearn.model_selection import train_test_split


file_path = './test/2020877-serie.txt'
data = read_dataset(file_path)


hyperparams = h = {
  'window_size': 6,
  'steps_forward': 1,
  'nb_sets': 4,
  'aggregation_opt': {'and_func':np.fmin,'or_func': np.fmax},
  'defuzzify_method': 'mom'
}

not_done = True
while (not_done):
  try:
    set_points = {
      'left_shoulder': None,
      'right_shoulder': None,
      'triangles': []
    }
    for _ in range(h['nb_sets']):
      min_val = 0.8 * data.min()
      max_val = 1.2 * data.max()
      d = ( max_val - min_val ) / 3
      step1 = min_val + d
      step2 = min_val + (2 * d)
      s = [random.triangular(min_val, step1), random.triangular(step1, step2), random.triangular(step2, max_val)]
      s.sort()
      set_points['triangles'].append(s) 

    res = evaluate_from_hyperparams(hyperparams, data, data, True, {'shoulder': False, 'set_points': set_points})
    next(res['sim'].ctrl.antecedents).view()
    not_done = False
  except Exception as e:
    pass    

print('-'*21 + '\nDataset Distribution\n' + '-'*21)
print('X\t:{} | y_true\t:{}'.format(res['datasets']['X'].shape, res['datasets']['y_true'].shape))
print('X_train\t:{} | y_train\t:{}'.format(res['datasets']['X_train'].shape, res['datasets']['y_train'].shape))
print('X_test\t:{}  | y_test\t:{}'.format(res['datasets']['X_test'].shape, res['datasets']['y_test'].shape))

print('\nmean_squared_error: {}'.format(res['out'][0]))
print('defuzzify method:', res['defuzzify_method'])

# plt.figure(figsize=(15,5))
# plt.plot(res['datasets']['y_true'], '*')
# plt.plot(res['out'][1], 's')
# plt.show()