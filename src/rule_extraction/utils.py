import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from typing import Callable, Dict
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split

from .fuzzy_rules import extract_rules, define_input_variables, define_output_variables



def generate_train_set_dataset(test_size, input_series, window_size, output_series, steps_forward):
  
  # Input
  idataset = tf.data.Dataset.from_tensor_slices(input_series)
  idataset = idataset.window(window_size + steps_forward, shift=1, drop_remainder=True)
  idataset = np.stack([list(window_dataset) for window_dataset in idataset], axis=0)

  # Output
  odataset = tf.data.Dataset.from_tensor_slices(output_series)
  odataset = odataset.window(window_size + steps_forward, shift=1, drop_remainder=True)
  odataset = np.stack([list(window_dataset) for window_dataset in odataset], axis=0)

  X, y_true = idataset[:,:-steps_forward,0], odataset[:,-steps_forward:][:,-1,:]
  X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=test_size, shuffle=False)

  return {
      'input': {'full': X, 'train': X_train, 'test': X_test},
      'output': {'full': y_true, 'train': y_train, 'test': y_test}
  }


def set_rules_configurations(
    nb_inputs: int,
    nb_outputs: int,
    nb_sets: int,
    x_min_value: float, 
    x_max_value: float, 
    y_min_value: float,
    y_max_value: float,
    aggregation_opt: Dict[str, Callable],
    defuzzify_method='centroid',
    resolution=1000,
    epsilon=0.0001,
    shoulder=False
):

  """
    Configuration
    -------------
      Variable Parameters
        - nb_inputs: Number of input variable
        - nb_outputs: Number of output variables
        - nb_sets: Number of fuzzy set by variable
        - min : list of min value for each variables
        - max : list of max value for each variables
        - shoulder : The first and last set are trapezoidal function

      Fuzzy System Parameters
        - defuzzify_method: Controls which defuzzification method will be used. 
            - 'centroid': Centroid of area 
            - 'bisector': bisector of area 
            - 'mom' : mean of maximum 
            - 'som' : min of maximum 
            - 'lom' : max of maximum
        - aggregation_opt: mathematic function for OR and AND operation

      Other Parameters (optional)
        - resolution: number of points in discret domain for each I/O variable.
        - epsilon: gap for min value (min - epsilon) and max value (max + epsilon)
  """
  config = {
    # Variable Parameters
    'nb_inputs' : nb_inputs,
    'nb_outputs': nb_outputs,
    'nb_sets': [[nb_sets] * nb_inputs, [nb_sets + 1] * nb_outputs],
    'min': [x_min_value,y_min_value],
    'max': [x_max_value,y_max_value],
    'shoulder': shoulder,

    # Fuzzy System Parameters
    'defuzzify_method': defuzzify_method,
    'aggregation_opt': aggregation_opt,

    #Other Parametres
    'resolution': resolution,
    'epsilon': epsilon
  }
  return config

def generate_sets(config, set_points=None):
  """
  Use set_rules_configuration() helper function for setting up the config input in the 
  correct format from the user defined hyperparameters.
  
  See: set_rules_configuration()
  """
  antecedents = define_input_variables(config, config['shoulder'], set_points)
  consequents = define_output_variables(config, config['shoulder'], config['defuzzify_method'], set_points)

  return antecedents, consequents



def evaluate_model(sim, x_data, y_data, return_results=False):
  y_prev = []
  for x in x_data:
      for i, x_i in enumerate(x,1):
          sim.input['I_{}'.format(i)] = x_i
      sim.compute()
      result = sim.output['O_1']
      y_prev.append(result)
      
  mse = mean_squared_error(y_data, y_prev)

  y_data, y_prev = np.squeeze(np.array(y_data)), np.array(y_prev)
  table_results = pd.DataFrame()
  table_results['Real']=y_data
  table_results['Predicted'] = y_prev
  table_results['Diference'] = np.abs(y_data - y_prev) 
  table_results['Diference (%)'] = np.abs((y_data - y_prev) / y_data)

  if return_results:
    return mse, y_prev, table_results
  return mse, y_prev,


def evaluate_from_hyperparams(
  hyperparams,
  input_series,
  output_series,
  out_datasets=False,
  options=None,
):
  """
  Arguments
  ---------

    hyperparams: {
      'window_size': int,
      'steps_forward': int,
      'nb_sets': int,
      'aggregation_opt': {'and_func': Callable,'or_func': Callable} 
    }

    options: {
      'test_size': float
      'defuzzify_method': str,
      'resolution': int,
      'epsilon': float
      'shoulder': bool
      'set_points': {
        left_shoulder: list[abcd]
        right_shoulder: list[abcd]
        triangles: list[][abc]
      }
    }

    Returns
    -------
      {
        'sim': sim,
        'window_size': h['window_size'],
        'nb_sets': h['nb_sets'],
        'aggregation_opt': h['aggregation_opt'],
        'datasets': h['datasets'] if out_datasets else None,
        'out': (mse, y_prev, table_results)
      }
  """

  h = hyperparams
  h['datasets'] = {'X': None, 'y_true': None, 'X_train': None,' X_test': None, 'y_train': None, 'y_test': None }

  _default_opts = {
    'test_size': 0.20,
    'defuzzify_method': 'centroid',
    'resolution': 1000,
    'epsilon': 0.0001,
    'shoulder': False,
    'set_points': None
  }
  options = {
      **_default_opts,
      **options
    } \
    if options != None else _default_opts 


  # Creating 80/20 train and test dataset
  datasets = generate_train_set_dataset(options['test_size'], input_series, h['window_size'], output_series, h['steps_forward'])

  h['datasets']['X'], h['datasets']['y_true'] = datasets['input']['full'], datasets['output']['full']
  h['datasets']['X_train'], h['datasets']['X_test'], h['datasets']['y_train'], h['datasets']['y_test']  = \
    datasets['input']['train'], datasets['input']['test'], \
    datasets['output']['train'], datasets['output']['test']

  # Hyperparams
  x_min_value = h['datasets']['X'].min(axis=0)
  x_max_value = h['datasets']['X'].max(axis=0)
  y_min_value = h['datasets']['y_true'].min(axis=0)
  y_max_value = h['datasets']['y_true'].max(axis=0)
  nb_inputs = h['window_size']
  nb_outputs = h['steps_forward']
  nb_sets = h['nb_sets']
  aggregation_opt= h['aggregation_opt']
  
  # Configuration
  config = set_rules_configurations(
      nb_inputs,
      nb_outputs,
      nb_sets,
      x_min_value,
      x_max_value,
      y_min_value, 
      y_max_value, 
      aggregation_opt,
      options['defuzzify_method'],
      options['resolution'],
      options['epsilon'],
      options['shoulder'],
  )
  # Sets
  antecedents, consequents = generate_sets(config, options['set_points'])
  # Rules extraction
  rules, df_rules = extract_rules(config, antecedents, consequents, h['datasets']['X_train'], h['datasets']['y_train'])
  # Control System
  system = ctrl.ControlSystem(rules)
  sim = ctrl.ControlSystemSimulation(system)
  # Evaluation
  mse, y_prev, table_results = evaluate_model(sim, h['datasets']['X'], h['datasets']['y_true'], return_results=True)

  return {
    'sim': sim,
    'window_size': h['window_size'],
    'nb_sets': h['nb_sets'],
    'aggregation_opt': h['aggregation_opt'],
    'datasets': h['datasets'] if out_datasets else None,
    'out': (mse, y_prev, table_results)
  }
