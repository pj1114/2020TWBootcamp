from knlm import KneserNey
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

class grammarErrorCorrector:
    def __init__(self, label_map_path, ngram_model_path, pos_model_path):
        
        self.label_map_path = label_map_path
        self.ngram_model_path = ngram_model_path
        self.pos_model_path = pos_model_path
        self.replacements = []
        self.mdl_2 = None
        self.pos = None
        self._initialize(self.label_map_path, self.ngram_model_path, self.pos_model_path)
        
    def _initialize(self, label_map_path, ngram_model_path, pos_model_path):
        with open(label_map_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                self.replacements.append( line.split('\t')[-1].replace('\n', '') )
        self.mdl_2 = KneserNey.load(ngram_model_path)
        self.pos = POS(POS_model_path)
        
    def redundantWordRemoval(self, input_text):
        poss = self.pos(input_text)
        text = list(input_text)
        deleted_candidates = []
        for i, ele in enumerate(text):
            if poss[i][0] != 'Na': 
                copied = text.copy()
                del copied[i]
                deleted_candidates.append((ele, ''.join(copied), self.mdl_2.evaluateSent(copied)))
        sorted_deleted_candidates = sorted(deleted_candidates, key=lambda k: k[2], reverse=True)
        return [ele[1] for ele in sorted_deleted_candidates[:3]]

    def missingWordAddition(self, input_text):
        text = list(input_text)
        res = []
        for i, ele in enumerate(text):
            for rep in self.replacements:
                copied = text.copy()
                copied.insert(i+1, rep)
                if self.mdl_2.evaluateSent(copied) > self.mdl_2.evaluateSent(text):
                    res.append(''.join(copied))
        return res