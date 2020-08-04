import numpy as np
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from gensim.models.word2vec import Word2Vec


class wordSub():
	def __init__(self, ws_model_path, pos_model_path, w2v_model_path):
		self.ws = WS(ws_model_path)
		self.pos = POS(pos_model_path)
		self.model = Word2Vec.load(w2v_model_path)
		self.w2v_results = []
		self.PIC_results = []

	def _PIC(self, target, sub, context, z1=2, z2=2):
		p_st = np.dot(sub.T,target)
		p_sc = sum([np.dot(sub.T,i) for i in context])/3
		pic = p_st + p_sc
		return pow(p_st*p_sc, 0.5)

	def get_word_subs(self, input_text):
		word_sentence_list = self.ws([input_text])
		pos_sentence_list = self.pos(word_sentence_list)[0]
		word_sentence_list = word_sentence_list[0]

		replaced_candidates = []
		context = [self.model[w] for w in word_sentence_list if w in self.model]
		for idx, word in enumerate(word_sentence_list):
			if 'V' in pos_sentence_list[idx]:
				replaced_candidates.append(word)
		for word in replaced_candidates:
			if word in self.model.wv.vocab:
				candidates = [ele[0] for ele in self.model.most_similar([word])]
				cand_score_dict = dict()
				for sub in candidates:
					cand_score_dict[sub]=self._PIC(self.model[word], self.model[sub], context)
				# word2vec's results
				self.w2v_results.append((word, candidates))
				# PIC metric's results
				self.PIC_results.append((word, sorted(candidates, key=lambda k: cand_score_dict[k], reverse=True)))
		return self.w2v_results, self.PIC_results 