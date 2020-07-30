import yaml
import os
from zailagan import ZaiLaGan
from metrics.spelling_error_metrics import SpellingErrorMetrics
import pickle
import traceback

# Load configuration file
with open("./config.yml", "r") as config_file_yaml:
  config = yaml.load(config_file_yaml, Loader = yaml.BaseLoader)

# Concatenate root path with all subpaths
root_path = config["User"]["user_path"]
model_names = ["gpt2_chinese", "ner"]
data_names = ["pinyin", "stroke", "dictionary", "common_char_set", "confusion", "place", "person", "ssc"]
for model_name in model_names:
	config["Model"][model_name] = os.path.join(root_path, config["Model"][model_name])
for data_name in data_names:
	config["Data"][data_name] = os.path.join(root_path, config["Data"][data_name])

# Instantiate ZaiLaGan
ZLG = ZaiLaGan(config)

# Instantiate SpellingErrorMetrics
SEM = SpellingErrorMetrics()

# Load data
path = "./data/spelling_error/sighan_2013.pkl"
file = open(path, "rb")
pairs = pickle.load(file)

# Calculate metrics
y_true = []
y_pred = []
for pair in pairs:
	try:
		wrong, correct, err_positions_true = pair[0], pair[1], pair[2]
		# Perform named-entity recognition first
		ner_processed_text, ne_positions = ZLG.detectNamedEntity([wrong])[0]
		ne_positions = set(ne_positions)
		# Detect spelling errors
		err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, 8e-3)
		# Remove potential spelling errors included in any named-entity
		err_positions_pred = []
		for err_position in err_positions:
			if(err_position not in ne_positions):
				err_positions_pred.append(err_position)
		y_true.append(err_positions_true)
		y_pred.append(err_positions_pred)
	except:
		traceback.print_exc()

# Show the results
print("False Alarm Rate: ", SEM.false_alarm_rate(y_pred, y_true))
print("Detection Accuracy: ", SEM.detection_accuracy(y_pred, y_true))
print("Detection Recall: ", SEM.detection_recall(y_pred, y_true))
print("Detection Precision: ", SEM.detection_precision(y_pred, y_true))
print("Detection F1: ", SEM.detection_f1(y_pred, y_true))
print("Error Location Accuracy: ", SEM.error_location_accuracy(y_pred, y_true))
print("Error Location Precision: ", SEM.error_location_precision(y_pred, y_true))
print("Error Location Recall: ", SEM.error_location_recall(y_pred, y_true))
print("Error Location F1: ", SEM.error_location_f1(y_pred, y_true))