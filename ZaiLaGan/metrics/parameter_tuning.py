from operator import itemgetter

class ParameterTuner:
  def __init__(self, metric_func, main_func, testing_data, param_candidates, y_true):
    self.metric_func = metric_func
    self.main_func = main_func
    self.param_candidates = param_candidates
    self.testing_data = testing_data
    self.y_true = y_true
      
  def evaluate_candidates(self):
    scores = []
    for idx in range(len(self.param_candidates)):
        y_pred = []
        param = self.param_candidates[idx]
        for text in self.testing_data:
            y_pred.append(self.main_func(text, param))
        scores.append(self.metric_func(y_pred, self.y_true))
    max_index, _ = max(enumerate(scores), key=itemgetter(1))
    return self.param_candidates[max_index]