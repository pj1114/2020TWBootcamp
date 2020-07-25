from typing import *
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from .trie import Trie
import codecs

class Utils:
	# Initialize config, device, model, and tokenizer
	def __init__(self, config):
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gpt2_model = GPT2LMHeadModel.from_pretrained(self.config["Model"]["gpt2_chinese"])
		self.gpt2_model.eval()
		self.gpt2_model = self.gpt2_model.to(self.device)
		self.bert_tokenizer = BertTokenizer(vocab_file = self.config["Model"]["gpt2_chinese"] + "/vocab.txt")

	# Load word dictionary (with word frequency if specified)
	def loadDictionary(self, path: str, hasFrequency = False) -> Dict[str,int]:
		word_dict = {}
		with open(path, "r") as dict_file:
			if(hasFrequency):
				for word in dict_file:
					word = word.split()
					if(word[0] in word_dict):
						word_dict[word[0]] += word[1]
					else:
						word_dict[word[0]] = word[1]
			else:
				for word in dict_file:
					word_dict[word] = -1
		return word_dict

	# Load word dictionary and convert it into trie (with word frequency if specified)
	def loadDictionaryTrie(self, path: str, hasFrequency = False):
		trie = Trie()
		dictionary = self.loadDictionary(path, hasFrequency)
		for word, frequency in dictionary.items():
			trie.addWord(word, frequency)

	# Load token-level similar stroke dictionary
	def loadStroke(self, path: str) -> Dict[str,List[str]]:
		stroke_dict = {}
		with open(path, "r") as stroke_file:
			for line in stroke_file:
				line = line.replace(" ", "").replace("\t", "").replace("\n", "")
				stroke_dict[line[0]] = list(line[1:])
		return stroke_dict

	# Load token-level similar pinyin dictionary
	def loadPinYin(self, path: str) -> Dict[str,List[str]]:
		pinyin_dict = {}
		with open(path, "r") as pinyin_file:
			for line in pinyin_file:
				line = line.replace(" ", "").replace("\t", "").replace("\n", "")
				pinyin_dict[line[0]] = list(line[1:])
		return pinyin_dict

	# Load place's named-entity dictionary
	def loadPlace(self, path: str) -> List[str]:
		place_lst = []
		with codecs.open(path, 'r', encoding = 'utf-8') as place_file:
			for line in place_file:
				line = line.strip()
				place = line.split('\t')[0]
				place_lst.append(place)
		return place_lst

	# Load person's named-entity dictionary
	def loadPerson(self, path: str) -> List[str]:
		person_lst = []
		with codecs.open(path, 'r', encoding = 'utf-8') as person_file:
			for line in person_file:
				line = line.strip()
				person = line.split('\t')[0]
				person_lst.append(person)
		return person_lst

	# Compute the perplexity of a sentence with language model
	def getSentencePpl(self, sentence: str) -> float:
		# Tokenize input sentence
		tokenized_sentence = self.bert_tokenizer.tokenize(sentence)
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)])
		token_ids = token_ids.to(self.device)
		# Compute perplexity
		with torch.no_grad():
			outputs = self.gpt2_model(token_ids, labels = token_ids)
			loss = outputs[0]
			return pow(2, loss.item())