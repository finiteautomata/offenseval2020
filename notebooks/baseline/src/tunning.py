def hyperparameter_tunning( params, num_param, current_combination):
  if num_param == len(params):
    print('entrenar')
    print(current_combination)

  else:
    print("num_param", num_param)
    key = list(params.keys())[num_param]
    values = list(params.values())[num_param]
    print(values)
    print(key)
    for val in values:
      current_combination[key] = val
      hyperparameter_tunning(params, num_param + 1, current_combination)