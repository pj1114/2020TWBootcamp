import numpy as np
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from gensim.models.word2vec import Word2Vec
import opencc
import nltk
from nltk.corpus import wordnet as wn

class wordSub():
	def __init__(self, ws_model_path, pos_model_path, w2v_model_path, anti_dict_path):
		nltk.download('wordnet')
		nltk.download('omw')

		self.ws = WS(ws_model_path)
		self.pos = POS(pos_model_path)
		self.model = Word2Vec.load(w2v_model_path)

		self.new_anti = self.build_antidict(anti_dict_path)
		self.cc1 = opencc.OpenCC('t2s')
		self.cc2 = opencc.OpenCC('s2t')
		

	def build_antidict(self, file):
		antidict = {}
		for line in open(file):
			line = line.strip().split(':')
			wd = line[0]
			antis = line[1].strip().split(';')
			if wd not in antidict:
				antidict[wd] = antis
			else:
				antidict[wd] += antis
		return antidict

	def sys_dict(self, word):
		antonyms = []
		synonyms = []
		for synset in wn.synsets(self.cc1.convert(word), lang='cmn'):
			synonyms.extend(synset.lemma_names('cmn'))
			for l in synset.lemmas():  
				if l.antonyms(): 
					antonyms.extend(l.antonyms()[0].synset().lemma_names('cmn'))
		synonyms = [self.cc2.convert(i) for i in set(synonyms)]
		antonyms = [self.cc2.convert(i) for i in set(antonyms)]
		return synonyms, antonyms

	def _getScore(self, sent1, sent2, model):
		return model.wmdistance(sent1, sent2)

	def get_word_subs(self, input_text):
		word_sentence_list = self.ws([input_text])
		pos_sentence_list = self.pos(word_sentence_list)[0]
		word_sentence_list = word_sentence_list[0]

		results = []

		for idx, word in enumerate(word_sentence_list):
			if 'V' in pos_sentence_list[idx]:
				if word in self.model.wv.vocab:
					if word in self.new_anti.keys():
						anti_word = self.new_anti[word]
					else:
						anti_word = []

				candidates = [ele[0] for ele in self.model.most_similar(positive=[word])]
				candidates = [ele for ele in candidates if 'V' in self.pos([[ele]])[0][0] 
								and ele not in self.sys_dict(word)[1]
									and ele not in anti_word]

				cand_score_dict = dict()
				for sub in candidates:
					word_sentence_list_copied = word_sentence_list.copy()
					word_sentence_list_copied[idx] = sub
					cand_score_dict[sub]=self._getScore(''.join(word_sentence_list), ''.join(word_sentence_list_copied), self.model)
				sorted_candidates = sorted([(i, cand_score_dict[i]) for i in candidates], key=lambda k:k[1])

				max_score, min_score = 0,0
				if len(sorted_candidates)!=0:
					max_score = sorted_candidates[-1][1]
					min_score = sorted_candidates[0][1]
				new_candidates = [(i[0],(cand_score_dict[i[0]]-min_score)/(max_score-min_score)) for i in sorted_candidates
									if 0.0< (cand_score_dict[i[0]]-min_score)/(max_score-min_score) < 0.5]

				results.append((word, new_candidates))

		return results 