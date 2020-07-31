import numpy as np
import pandas as pd

# SpellingErrorMetrics class
class SpellingErrorMetrics():
  def __init__(self):
    self.name = 'SpellingErrorMetrics'
      
  # Error detection    
  # Input detected error locations and true error locations
  def false_alarm_rate(self, y_pred, y_true):
    y_pred_1 = [1 if len(i) != 0 else 0 for i in y_pred]
    y_true_1 = [1 if len(i) != 0 else 0 for i in y_true]
    return np.mean([i != j for i, j in zip(y_pred_1, y_true_1) if j == 0])

  def detection_accuracy(self, y_pred, y_true):
    y_pred_1 = [1 if len(i) != 0 else 0 for i in y_pred]
    y_true_1 = [1 if len(i) != 0 else 0 for i in y_true]
    return np.mean([i == j for i, j in zip(y_pred_1, y_true_1)])

  def detection_recall(self, y_pred, y_true):
    y_pred_1 = [1 if len(i) != 0 else 0 for i in y_pred]
    y_true_1 = [1 if len(i) != 0 else 0 for i in y_true]
    return np.mean([i == j for i, j in zip(y_pred_1, y_true_1) if j != 0])

  def detection_precision(self, y_pred, y_true):
    y_pred_1 = [1 if len(i) != 0 else 0 for i in y_pred]
    y_true_1 = [1 if len(i) != 0 else 0 for i in y_true]
    return np.mean([i == j for i, j in zip(y_pred_1, y_true_1) if i != 0])

  def detection_f1(self, y_pred, y_true):
    recall = self.detection_recall(y_pred, y_true)
    precision = self.detection_precision(y_pred, y_true)
    return (2*recall*precision) / (recall+precision)

  def error_location_accuracy(self, y_pred, y_true):
    TP_TN = 0
    for i, j in zip(y_pred, y_true):
      if i == j:
        TP_TN += 1
    return TP_TN / len(y_pred)

  def error_location_precision(self, y_pred, y_true):
    TP, FP = 0, 0
    for i, j in zip(y_pred, y_true):
      if len(i) > 0:
        if i == j:
          TP += 1
        else: 
          FP += 1
    return TP / (TP+FP)

  def error_location_recall(self, y_pred, y_true):
    TP, FN = 0, 0
    for i, j in zip(y_pred, y_true):
      if len(j) > 0:
        if i == j:
          TP += 1
        else: 
          FN += 1
    return TP / (TP+FN)

  def error_location_f1(self, y_pred, y_true):
    ELP = self.error_location_precision(y_pred, y_true)
    ELR = self.error_location_recall(y_pred, y_true)
    return (2*ELP*ELR) / (ELP+ELR)
      
  # Error correction    
  # Input corrected sentences
  def correction_accuracy(self, y_pred, y_true):
    return np.mean([i == j for i, j in zip(y_pred, y_true)])

  def correction_precision(self, y_original, y_pred, y_true):
    y_change = [i == j for i, j in zip(y_original, y_pred)]
    return np.mean([i == j for i, j, k in zip(y_pred, y_true, y_original) if k != 0])