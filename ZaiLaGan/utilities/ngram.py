import json
import math
from knlm import KneserNey

class NGRAM():
	def __init__(self, model_path):
		self.model_path = model_path
		self.model = KneserNey.load(self.model_path)
	def get_ppl(self, sentence):
		L = self.model.evaluateSent(sentence)
    	return pow(math.exp(1), -L/len(sentence))