from operator import itemgetter
from itertools import groupby
from skfuzzy import control as ctrl

import numpy as np
import skfuzzy as fuzz

def read_dataset(file_path):
    def preprocessing(line):
        values = line.strip().replace(',','.').split('\t')
        return [float(v) for v in values]

    with open(file_path,'r') as file:
        series = [preprocessing(line) for line in file]
    return np.array(series)

def config_input_variable(name, ns, min_val, max_val, resolution, shoulder, set_points=None):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Antecedent(universe, name)
  var = config_set_variable(var, ns, min_val, max_val, shoulder, set_points)
  return var

def config_output_variable(name, ns, min_val, max_val, resolution, shoulder, defuzzify_method, set_points=None):
  universe = np.linspace(min_val, max_val, resolution)
  var = ctrl.Consequent(universe, name, defuzzify_method=defuzzify_method)
  var = config_set_variable(var, ns, min_val, max_val, shoulder, set_points)
  return var

def config_set_points(ns, min_val, max_val, shoulder):
  points = {}
  if shoulder:
      c = np.linspace(min_val, max_val, ns+2)
      d = ( max_val - min_val ) / (ns + 1)

      ## First Shoulder
      points['s_1'] = [c[0]-d, c[0] , c[0]+d, c[0]+2*d]
      # Triangles
      for s in range(2,ns):
        points['s_{}'.format(s)] = [c[s-1], c[s], c[s+1]]

      # Right Shoulder
      points['s_{}'.format(ns)] = [c[ns]-d , c[ns], c[ns]+d, c[ns] + 2*d]
      return points
  else:
      c = np.linspace(min_val, max_val, ns)
      d = ( max_val - min_val ) / (ns - 1)
      for s in range(1,ns+1):
        points['s_{}'.format(s)] = [c[s-1] -d, c[s-1], c[s-1] + d]
      return points


def config_set_variable(var, ns, min_val, max_val, shoulder, set_points=None):
  c = None
  if shoulder:
      if( set_points != None and _validate_points(set_points, ns)):
        c = set_points
        var['s_1'] = fuzz.trapmf(var.universe, c['left_shoulder'])
        for i, s in enumerate(c['triangles']):
          var['s_{}'.format(i + 1)] = fuzz.trimf(var.universe, s)
        var['s_{}'.format(ns)] = fuzz.trapmf(var.universe, c['right_shoulder'])
      else:
        c = np.linspace(min_val, max_val, ns+2)
        d = ( max_val - min_val ) / (ns + 1)
        var['s_1'] = fuzz.trapmf(var.universe, [c[0]-d, c[0] , c[0]+d, c[0]+2*d])
        for s in range(2,ns):
          var['s_{}'.format(s)] = fuzz.trimf(var.universe, [c[s-1], c[s], c[s+1]])
        var['s_{}'.format(ns)] = fuzz.trapmf(var.universe, [c[ns]-d , c[ns], c[ns]+d, c[ns] + 2*d])
      return var
  else:
      if( set_points != None and _validate_points(set_points, ns)):
        c = set_points
        for i, s in enumerate(c['triangles']):
          var['s_{}'.format(i + 1)] = fuzz.trimf(var.universe, s)
      else:
        c = np.linspace(min_val, max_val, ns)
        d = ( max_val - min_val ) / (ns - 1)
        for s in range(1,ns+1):
          var['s_{}'.format(s)] = fuzz.trimf(var.universe, [c[s-1] -d, c[s-1], c[s-1] + d])
      return var

def define_input_variables(config, shoulder, set_points=None):
  epsilon = config['epsilon']
  input_variables  = [config_input_variable('I_{}'.format(i+1), 
  config['nb_sets'][0][i], 
  config['min'][0][i]-epsilon, 
  config['max'][0][i]+epsilon,
  config['resolution'], 
  shoulder,
  set_points) \
                    for i in range(config['nb_inputs'])]
  return input_variables

def define_output_variables(config, shoulder, defuzzify_method, set_points=None):
  epsilon = config['epsilon']
  output_variables = [config_output_variable('O_{}'.format(i+1), 
  config['nb_sets'][1][i], 
  config['min'][1][i]-epsilon, 
  config['max'][1][i]+epsilon, 
  config['resolution'], 
  shoulder, 
  defuzzify_method,
  set_points) \
                    for i in range(config['nb_outputs'])]
  return output_variables

def extract_rules(config, input_variables , output_variables, X, y):
  variables = input_variables + output_variables
  data = np.concatenate([X,y],axis=1)
  all_rule_candidate = []
  for instance in data:
    rule_candidate = []
    for x,var in zip(instance,variables):
      u_x_s = [fuzz.interp_membership(var.universe, var[term].mf, x)  for term in var.terms]
      term_max = np.argmax(u_x_s)
      rule_candidate.append((term_max , u_x_s[term_max]))
    us = [u for t,u in rule_candidate]
    dr  = np.prod(us) 
    key = '_'.join([str(t) for t,u in rule_candidate[:-1]]) 
    all_rule_candidate.append((key, dr, rule_candidate))
  
  sorted_all_rule_candidate = sorted(all_rule_candidate, key=itemgetter(0))
  groups = groupby(sorted_all_rule_candidate, key=itemgetter(0))
  ant_con = []
  for k,v in groups:
    _,dr,rule_candidate = zip(*v)
    choosen = np.argmax(dr)
    rule = rule_candidate[choosen]
    dr_val = dr[choosen]
    terms,_ = zip(*rule)
    ant = terms[:-1]
    con = terms[-1]
    ant_con.append((ant,con, dr_val))

  rules = []
  rules_view = []
  for ant,con,dr_val in ant_con:
    ant_vars = []
    for a,var in zip(ant,input_variables):
      terms = [t for t in var.terms]
      ant_vars.append(var[terms[a]])
    rule_str=' &  '.join(['ant_vars[{}]'.format(i) for i in range(len(ant_vars))])
    antc = eval(rule_str)

    var = output_variables[-1]
    terms = [t for t in var.terms]
    cons = var[terms[con]]
    rules.append(ctrl.Rule(antc, cons, **config['aggregation_opt']))
    rules_view.append((antc, cons, dr_val))
  return rules

# def rules_dataframe(rules_view):
#   for rule in rules_view:
#       dr_val = rule[2]
#       con = rule[1]
#       t1=rule[0]
#       ts=[]
#       while True:
#           try:
#               t2=t1.term2
#               ts.append(t2)
#               t1=t1.term1        
#           except:
#               ts.append(t1)
#               break
#       order_columns = ['I_{}'.format(i+1) for i,a in enumerate(ts[::-1])] + ['O_1', 'Dr']
#       rule_dict = {'I_{}'.format(i+1):a.label for i,a in enumerate(ts[::-1])}
#       rule_dict['O_1']= con.label
#       rule_dict['Dr']= dr_val
#       df=df.append(rule_dict, ignore_index=True)
#       df = df[order_columns]
#   return df

def _validate_points(set_points, nb_sets):
  """
    Parameters
    ----------

      set_points: {
        left_shoulder: list[abcd]
        right_shoulder: list[abcd]
        triangles: list[][abc]
      }
  """
  valid = True
  if(set_points['left_shoulder'] == None or set_points['right_shoulder'] == None):
    valid = len(set_points['triangles']) == nb_sets
    for s in set_points['triangles']:
      valid = s[0] <= s[1] and s[1] <= s[2] 
  else:
    valid = \
      set_points['left_shoulder'][0] <= set_points['left_shoulder'][1] and \
      set_points['left_shoulder'][1] <= set_points['left_shoulder'][2] and \
      set_points['left_shoulder'][2] <= set_points['left_shoulder'][3] and \
      set_points['right_shoulder'][0] <= set_points['right_shoulder'][1] and \
      set_points['right_shoulder'][1] <= set_points['right_shoulder'][2] and \
      set_points['right_shoulder'][2] <= set_points['right_shoulder'][3]

    valid = len(set_points['triangles']) == nb_sets - 2
    for s in set_points['triangles']:
      valid = s[0] <= s[1] and s[1] <= s[2] 
  return valid