from typing import *
import torch
from transformers import BertTokenizer, BertForMaskedLM
from utilities.utils import Utils
from utilities.ner import *
import re
from pypinyin import lazy_pinyin

class ZaiLaGan():
  # Initialize config, device, model, tokenizer, and utilities
  def __init__(self, config):
    self.config = config
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.bert_wwm_model = BertForMaskedLM.from_pretrained(self.config["Model"]["bert_wwm_ext_chinese"])
    self.bert_wwm_model.eval()
    self.bert_wwm_model = self.bert_wwm_model.to(self.device)
    self.bert_wwm_tokenizer = BertTokenizer.from_pretrained(self.config["Model"]["bert_wwm_ext_chinese"])
    self.bert_base_model = BertForMaskedLM.from_pretrained(self.config["Model"]["bert_base_chinese"])
    self.bert_base_model.eval()
    self.bert_base_model = self.bert_base_model.to(self.device)
    self.bert_base_tokenizer = BertTokenizer.from_pretrained(self.config["Model"]["bert_base_chinese"])
    self.utils = Utils(self.config)
    self.dict_trie = self.utils.loadDictionaryTrie(self.config["Data"]["dictionary"], True)
    self.pinyin = self.utils.loadPinYin(self.config["Data"]["pinyin"])
    self.stroke = self.utils.loadStroke(self.config["Data"]["stroke"])
    self.place = self.utils.loadPlace(self.config["Data"]["place"])
    self.person = self.utils.loadPerson(self.config["Data"]["person"]) 
    self.ner_model = NER(self.config["Model"]["ner"], self.pinyin, self.stroke, self.place, self.person, self.config["Data"]["ssc"])
    self.charSet = self.utils.loadCharSet(self.config['Data']['common_char_set'])
    self.customConfusionDict = self.utils.loadCustomConfusion(self.config['Data']['confusion'])

  # Detect named-entities and return their corrections & positions
  def detectNamedEntity(self, sentences: List[str]) -> List[Tuple[str,List[int]]]:
    return self.ner_model.check_ner(sentences)

  # Detect potential spelling errors in a given sentence/paragraph and return detected error positions & top predictions from BERT
  def detectSpellingError(self, text: str, threshold: float) -> Tuple[List[int],Dict[int,List[str]]]:
    positions = []
    predictions = {}
    # Mask each word and predict it
    for i in range(len(text)):
      # Check if current word is a chinese character
      if(not self.utils.isChineseChar(text[i])):
        continue
      # Add mask
      masked_text = "[CLS]" + text[:i] + "[MASK]" + text[i+1:] + "[SEP]"
      # Tokenize input text
      tokenized_masked_text = self.bert_wwm_tokenizer.tokenize(masked_text)
      masked_token_index = tokenized_masked_text.index("[MASK]")
      # Construct token ids and segment ids
      token_ids = torch.tensor([self.bert_wwm_tokenizer.convert_tokens_to_ids(tokenized_masked_text)])
      segment_ids = torch.tensor([[0] * token_ids.shape[1]])
      # Set up ids on GPU
      token_ids = token_ids.to(self.device)
      segment_ids = segment_ids.to(self.device)
      # Predict masked token
      with torch.no_grad():
        outputs = self.bert_wwm_model(token_ids, token_type_ids = segment_ids)
        scores = outputs[0][0,masked_token_index]
        # Classify the token as a potential spelling error if predicted probability is lower than given threshold
        token_probability = torch.nn.Softmax(0)(scores)[self.bert_wwm_tokenizer.convert_tokens_to_ids(text[i])]
        if(token_probability < threshold):
          # Extract top predictions from BERT
          token_scores, token_indices = scores.topk(5)
          top_predicted_tokens = self.bert_wwm_tokenizer.convert_ids_to_tokens(token_indices)
          positions.append(i)
          predictions[i] = top_predicted_tokens
    return (positions, predictions)

  # Give top n suggestions of spelling error correction
  def correctSpellingError(self, text: str, err_positions: Set[int], predictions: Dict[int,List[str]], ne_positions: Set[int], candidate_num: int) -> List[str]:
    # Initialize a dictionary to record starting positions of potentially correct tokens/words
    starting_positions = {}
    # Add original tokens
    for i in range(len(text)):
      token = text[i]
      starting_positions[i] = set(token)
    # Add similar tokens in stroke or pinyin
    for err_position in err_positions:
      # Check if the error token is included in a named-entity
      if(err_position in ne_positions):
        continue
      else:
        error_token = text[err_position]
        if(error_token in self.stroke):
          for similar_token in self.stroke[error_token][:5]:
            starting_positions[err_position].add(similar_token)
        if(error_token in self.pinyin):
          for similar_token in self.pinyin[error_token][:10]:
            starting_positions[err_position].add(similar_token)
        for predicted_token in predictions[err_position]:
          # Check if BERT's prediction is a chinese character
          if(len(predicted_token) == 1 and self.utils.isChineseChar(predicted_token)):
            starting_positions[err_position].add(predicted_token)
    # Construct candidate sentences
    candidates = []
    prefixes = list(starting_positions[0])
    while(len(prefixes) > 0):
      prefix = prefixes.pop(0)
      if(len(prefix) == len(text)):
        candidates.append((prefix,self.utils.getSentencePpl(prefix)))
      else:
        for suffix in starting_positions[len(prefix)]:
          prefixes.append(prefix+suffix)
    # Sort candidate sentences by perplexity and get top n suggestions
    candidates.sort(key = lambda x: x[1])
    recommendations = []
    for i in range(min(len(candidates),candidate_num)):
      recommendations.append(candidates[i][0])
    return recommendations

  def generate_correction_cand(self, word):
    correction_candidates = []
    if len(word) == 1:
      # Add similar tokens in pinyin
      confusion_word_set = set()
      for char in self.charSet:
        if lazy_pinyin(char) == lazy_pinyin(word):
          confusion_word_set.add(char)
      confusion_word_set = confusion_word_set.union(set(self.pinyin[word]))
      correction_candidates.extend(confusion_word_set)
      # Add similar tokens in stroke
      correction_candidates.extend(self.stroke[word])
        
    if len(word) > 2:
      edit_cand = set()
      word_splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
      transposes = [L + R[1] + R[0] + R[2:] for L, R in word_splits if len(R) > 1]
      replaces = [L + c + R[1:] for L, R in word_splits if R for c in self.charSet]
      edit_set = set(transposes + replaces)
      for edit in edit_set: 
        if self.dict_trie.getWordFreq(edit) > 0:
          edit_cand.add(edit)
      correction_candidates.extend(edit_cand)   
      
      confusion_word_set = set()
      if word in self.customConfusionDict:
        confusion_word_set = {self.customConfusionDict[word]}
      correction_candidates.extend(confusion_word_set)
      
      if len(word) == 2:
        # Add similar tokens in pinyin
        correction_candidates.extend(set(ele + word[1:] for ele in self.pinyin[word[0]] if ele))
        correction_candidates.extend(set(word[:-1]+ele for ele in self.pinyin[word[-1]] if ele))

      if len(word) > 2:
        correction_candidates.extend(set(word[0] + ele + word[2:] for ele in self.pinyin[word[1]] if ele))
        correction_candidates.extend(set(ele + word[-1] for ele in self.pinyin[word[1]] if ele))
        correction_candidates.extend(set(word[0] + ele for ele in self.pinyin[word[1]] if ele))
        
    return correction_candidates

  def bertDetectAndCorrect(self, text: str, topk: int):
    positions = []
    # split input text into short texts
    re_punctuation = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&]+)", re.U)
    short_texts = []
    components = re_punctuation.split(text)
    components = list(filter(('').__ne__, components))
    start_idx = 0
    for comp in components:
      if re_punctuation.match(comp):
        short_texts.append((comp, start_idx))
    start_idx += len(comp)
    
    # character-based detection and correction
    for (short_text, start_idx) in short_texts:
      for idx, single_word in enumerate(short_text):
        # bert-based model generates topk candidates 
        masked_text = "[CLS]" + text[:idx] + "[MASK]" + text[idx+1:] + "[SEP]"
        tokenized_masked_text = self.bert_base_tokenizer.tokenize(masked_text)
        token_ids = torch.tensor([self.bert_base_tokenizer.convert_tokens_to_ids(tokenized_masked_text)])
        segment_ids = torch.tensor([[0] * token_ids.shape[1]])
        token_ids = token_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        with torch.no_grad():
          outputs = self.bert_base_model(token_ids, token_type_ids = segment_ids)
          scores = outputs[0][0,idx+1]
          token_probability = torch.nn.Softmax(0)(scores)[self.bert_base_tokenizer.convert_tokens_to_ids(text[idx])]
          scores_list = torch.nn.Softmax(0)(scores)
          _, pred = scores_list.topk(topk, 0, True, True)
          topk_bert_candidates = [self.bert_base_tokenizer.convert_ids_to_tokens(ele.item()) for ele in pred]
            
        if topk_bert_candidates and (single_word not in topk_bert_candidates):
          candidates = self.generate_correction_cand(short_text[idx])
          candidates_sorted = sorted(candidates, key=lambda k: self.dict_trie.getWordFreq(k), reverse=True)
          if candidates_sorted:
            for topk_bert_cand in topk_bert_candidates:
              if topk_bert_cand in candidates_sorted:
                #print(['- '+single_word, '+ '+topk_bert_cand + '_'+str(start_idx+idx)])
                positions.append(start_idx+idx)
                single_word = topk_bert_cand
                break
                            
    # word-based detection and correction
    for (short_text, start_idx) in short_texts:
      for n in [2, 3, 4, 5]:
        for idx in range(len(short_text) - n + 1):
          word = short_text[idx: idx+n]
          # bert-based model generates topk candidates 
          masked_text = "[CLS]" + text[:idx] + "[MASK]" + text[idx+1:] + "[SEP]"
          tokenized_masked_text = self.bert_base_tokenizer.tokenize(masked_text)
          token_ids = torch.tensor([self.bert_base_tokenizer.convert_tokens_to_ids(tokenized_masked_text)])
          segment_ids = torch.tensor([[0] * token_ids.shape[1]])
          token_ids = token_ids.to(self.device)
          segment_ids = segment_ids.to(self.device)
          with torch.no_grad():
            outputs = self.bert_base_model(token_ids, token_type_ids = segment_ids)
            scores = outputs[0][0,idx+1]
            token_probability = torch.nn.Softmax(0)(scores)[self.bert_base_tokenizer.convert_tokens_to_ids(text[idx])]
            scores_list = torch.nn.Softmax(0)(scores)
            _, pred = scores_list.topk(topk, 0, True, True)
            topk_bert_candidates = [self.bert_base_tokenizer.convert_ids_to_tokens(ele.item()) for ele in pred]
          
          candidates = self.generate_correction_cand(word)
          candidates = [ele for ele in candidates if self.dict_trie.getWordFreq(ele)>0]
          if candidates:
            for topk_bert_cand in topk_bert_candidates:
              tmp_word = topk_bert_cand + word[1:]
              if tmp_word in candidates and tmp_word!= word:
                #print(['- '+short_text[idx], '+ '+topk_bert_cand + '_'+str(start_idx+idx)])
                positions.append(start_idx+idx)
                break
    return list(set(positions))