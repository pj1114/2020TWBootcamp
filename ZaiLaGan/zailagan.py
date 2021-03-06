from typing import *
import torch
from transformers import BertTokenizer, BertForMaskedLM
from utilities.utils import Utils
from utilities.ner import *
import re
from pypinyin import lazy_pinyin
from utilities.ngram import *
import subprocess
import os
import ast

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
    self.ngram_model = NGRAM(self.config["Model"]["ngram"])
    #self.GEC = grammarErrorCorrector(self.config["Data"]["label_map"], self.config["Model"]["ngram"], self.config["Model"]["pos_model"])

  # Detect named-entities and return their corrections & positions
  def detectNamedEntity(self, sentences: List[str], task_name:str) -> List[Tuple[str,List[int]]]:
    return self.ner_model.check_ner(sentences, task_name)

  # Detect potential spelling errors in a given sentence/paragraph and return detected error positions & top predictions from BERT
  def detectSpellingError(self, text: str, threshold: float, topk: int) -> Tuple[List[int],Dict[int,List[str]]]:
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
          token_scores, token_indices = scores.topk(topk)
          top_predicted_tokens = self.bert_wwm_tokenizer.convert_ids_to_tokens(token_indices)
          positions.append(i)
          predictions[i] = top_predicted_tokens
    return (positions, predictions)

  # Give top n suggestions of spelling error correction
  def correctSpellingError(self, text: str, err_positions: Set[int], predictions: Dict[int,List[str]], ne_positions: Set[int], candidate_num: int, similar_bonus: float) -> List[Tuple[str,int,float]]:
    # Initialize a dictionary to record starting positions of potentially correct tokens/words
    starting_positions = {}
    # Add original tokens
    for i in range(len(text)):
      token = text[i]
      # Separate all tokens/words from tokens/words that are similar in stroke or pinyin
      starting_positions[i] = (set(token), set(token))
    # Add similar tokens in stroke or pinyin
    for err_position in err_positions:
      # Check if the error token is included in a named-entity
      if(err_position in ne_positions):
        continue
      else:
        error_token = text[err_position]
        ssc_target = self.ner_model.ssc.getSoundCode(error_token)
        if(error_token in self.stroke):
          for similar_token in self.stroke[error_token][:3]:
            starting_positions[err_position][0].add(similar_token)
            starting_positions[err_position][1].add(similar_token)
        if(error_token in self.pinyin):
          for similar_token in self.pinyin[error_token][:7]:
            starting_positions[err_position][0].add(similar_token)
            starting_positions[err_position][1].add(similar_token)
        for predicted_token in predictions[err_position]:
          # Check if BERT's prediction is a chinese character
          if(len(predicted_token) == 1 and self.utils.isChineseChar(predicted_token)):
            starting_positions[err_position][0].add(predicted_token)
            ssc_pred = self.ner_model.ssc.getSoundCode(predicted_token)
            ssc_score = self.ner_model.ssc.computeSoundCodeSimilarity(ssc_target, ssc_pred)
            if ssc_score >= 0.7 :
              starting_positions[err_position][1].add(predicted_token)
    # Construct candidate sentences
    candidates = []
    prefixes = list(starting_positions[0][0])
    # Initialize counts of tokens/words that are similar in stroke or pinyin
    for i in range(len(prefixes)):
      if(prefixes[i] in starting_positions[0][1]):
        prefixes[i] = (prefixes[i], 1)
      else:
        prefixes[i] = (prefixes[i], 0)
    while(len(prefixes) > 0):
      prefix = prefixes.pop(0)
      if(len(prefix[0]) == len(text)):
        candidates.append((prefix[0],prefix[1],self.ngram_model.get_ppl(prefix[0])))
      else:
        for suffix in starting_positions[len(prefix[0])][0]:
          if(suffix in starting_positions[len(prefix[0])][1]):
            prefixes.append((prefix[0]+suffix,prefix[1]+1))
          else:
            prefixes.append((prefix[0]+suffix,prefix[1]))
    # Sort candidate sentences by perplexities from ngram model
    candidates.sort(key = lambda x: x[2])
    # Compute top candidate sentences' perplexities again with GPT2 and sort
    candidates = candidates[:50]
    for i in range(len(candidates)):
      candidates[i] = (candidates[i][0], candidates[i][1], self.utils.getSentencePpl(candidates[i][0]))
    candidates.sort(key = lambda x: x[2])
    # Extract top n suggestions
    recommendations = []
    for i in range(min(len(candidates),candidate_num)):
      recommendations.append(candidates[i])
    # Take counts of tokens/words that are similar in stroke or pinyin into consideration and sort again
    for i in range(len(recommendations)):
      recommendations[i] = (recommendations[i][0], recommendations[i][1], recommendations[i][2]/pow(similar_bonus,recommendations[i][1]))
    recommendations.sort(key = lambda x: x[2])
    return recommendations

  # Find spelling error correction candidates from dictionary
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

  # Detect and correct spelling errors given input text
  def bertDetectAndCorrect(self, text: str, topk: int, ner_pos_list: List[int]) -> Tuple[str, List[int]]:
    positions = []
    text_list = list(text)
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
        start_idx += 1
    # character-based detection and correction
    for (short_text, start_idx) in short_texts:
      for idx, single_word in enumerate(short_text):
        if self.utils.isChineseChar(single_word):
          if start_idx+idx not in ner_pos_list:
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
                    text_list[start_idx+idx] = topk_bert_cand
                    positions.append(start_idx+idx)
                    single_word = topk_bert_cand
                    break
                            
    # word-based detection and correction
    for (short_text, start_idx) in short_texts:
      for n in [2, 3, 4, 5]:
        for idx in range(len(short_text) - n + 1):
          if not ner_pos_list or (ner_pos_list and (start_idx+idx > ner_pos_list[-1] or start_idx+idx+n < ner_pos_list[0])):
            if self.utils.isChineseChar(short_text[idx]):
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
                    text_list[start_idx+idx] = topk_bert_cand
                    positions.append(start_idx+idx)
                    break
    # return corrected string and error position list
    return (''.join(text_list), sorted(list(set(positions))) )

  # Divide a long text into multiple parts and correct spelling errors separately
  def divideAndCorrectSpellingError(self, text: str) -> Tuple[str, str]:
    # Perform named-entity recognition first
    ner_processed_text, ne_positions, _ = self.detectNamedEntity([text], 'correction')[0]
    ne_positions = set(ne_positions)
    # Detect spelling errors
    err_positions, bert_predictions = self.detectSpellingError(ner_processed_text, 1e-5, 3)
    err_positions = set(err_positions)
    # Split long text into multiple parts
    punctuations = {"。", "?", "!", "，", "、", ";", ":"}
    splitted_text = []
    sub_ne_positions, sub_err_positions, sub_bert_predictions = set(), set(), {}
    start = 0
    count = 0
    for i in range(len(ner_processed_text)):
      # Check if current character is included in a named-entity or is an error
      if(i in ne_positions):
        sub_ne_positions.add(i-start)
      if(i in err_positions):
        sub_err_positions.add(i-start)
        sub_bert_predictions[i-start] = bert_predictions[i]
      # Check if current character is a punctuation
      if(ner_processed_text[i] in punctuations):
        count += 1
      # Check if a short text has been completed
      if(count == 2):
        splitted_text.append((ner_processed_text[start:i+1], sub_err_positions, sub_bert_predictions, sub_ne_positions))
        sub_ne_positions, sub_err_positions, sub_bert_predictions = set(), set(), {}
        start = i + 1
        count = 0
      elif(i == len(ner_processed_text) - 1):
        splitted_text.append((ner_processed_text[start:], sub_err_positions, sub_bert_predictions, sub_ne_positions))
    # Correct spelling errors in each short text and combine corrected results
    corrections = []
    for short_text in splitted_text:
      correction = self.correctSpellingError(short_text[0], short_text[1], short_text[2], short_text[3], 10, 1.5)[0][0]
      corrections.append(correction)
    return (ner_processed_text, "".join(corrections))
  
  # Get substitution words
  def getWordSub(self, text):
    res = os.popen("conda run -n wordSub python ./utilities/wordSubJob.py "+ text).read()
    dic = ast.literal_eval(res)
    return dic