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
model_names = ["gpt2_chinese", "ner", "ngram"]
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

# Calculate metrics for different detection thresholds
thresholds = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2]
results = {}
for threshold in thresholds:
	y_true = []
	y_pred = []
	# Iterate through each sentence to perform spelling error detection
	for pair in pairs:
		try:
			wrong, err_positions_true = pair[0], pair[1]
			# Perform named-entity recognition first
			ner_processed_text, ne_positions, ne_err_positions = ZLG.detectNamedEntity([wrong], "detection")[0]
			ne_positions = set(ne_positions)
			# Detect spelling errors
			err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, threshold, 5)
			# Remove potential spelling errors included in any named-entity
			err_positions_pred = []
			for err_position in err_positions:
				if(err_position not in ne_positions):
					err_positions_pred.append(err_position)
			y_true.append(err_positions_true)
			y_pred.append(err_positions_pred)
		except:
			traceback.print_exc()
	try:
		# Store the results
		result = {}
		result["false_alarm_rate"] = SEM.false_alarm_rate(y_pred, y_true)
		result["detection_accuracy"] = SEM.detection_accuracy(y_pred, y_true)
		result["detection_recall"] = SEM.detection_recall(y_pred, y_true)
		result["detection_precision"] = SEM.detection_precision(y_pred, y_true)
		result["detection_f1"] = SEM.detection_f1(y_pred, y_true)
		result["error_location_accuracy"] = SEM.error_location_accuracy(y_pred, y_true)
		result["error_location_precision"] = SEM.error_location_precision(y_pred, y_true)
		result["error_location_recall"] = SEM.error_location_recall(y_pred, y_true)
		result["error_location_f1"] = SEM.error_location_f1(y_pred, y_true)
		results[threshold] = result
		print(threshold, result)
	except:
		traceback.print_exc()

# Store the results in a pickle file
with open("./tune_result.pkl", "wb") as tune_result:
	pickle.dump(results, tune_result)