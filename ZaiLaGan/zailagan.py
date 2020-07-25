from typing import *
import torch
from transformers import BertTokenizer, BertForMaskedLM
from zhon import hanzi
from utilities.utils import Utils
from utilities.ner import *

class ZaiLaGan():
  # Initialize config, device, model, tokenizer, and utilities
  def __init__(self, config):
    self.config = config
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.bert_wwm_model = BertForMaskedLM.from_pretrained(self.config["Model"]["bert_wwm_ext_chinese"])
    self.bert_wwm_model.eval()
    self.bert_wwm_model = self.bert_wwm_model.to(self.device)
    self.bert_wwm_tokenizer = BertTokenizer.from_pretrained(self.config["Model"]["bert_wwm_ext_chinese"])
    self.utils = Utils(self.config)
    self.dict_trie = self.utils.loadDictionaryTrie(self.config["Data"]["dictionary"], True)
    self.pinyin = self.utils.loadPinYin(self.config["Data"]["pinyin"])
    self.stroke = self.utils.loadStroke(self.config["Data"]["stroke"])
    self.place = self.utils.loadPlace(self.config["Data"]["place"])
    self.person = self.utils.loadPerson(self.config["Data"]["person"]) 
    self.ner_model = NER(self.config["Model"]["ner"], self.pinyin, self.stroke, self.place, self.person, self.config["Data"]["ssc"])

  # Detect named-entities and return their corrections & positions
  def detectNamedEntity(self, sentences: List[str]):
    return self.ner_model.check_ner(sentences)

  # Detect potential spelling errors in a given sentence/paragraph and return detected error positions & top predictions from BERT
  def detectSpellingError(self, text: str, threshold: float):
    positions = []
    predictions = {}
    # Mask each word and predict it
    for i in range(len(text)):
      # Check if current word is a punctuation
      if(text[i] in hanzi.punctuation):
        continue
      # Add mask
      masked_text = "[CLS]" + text[:i] + "[MASK]" + text[i+1:] + "[SEP]"
      # Tokenize input text
      tokenized_masked_text = self.bert_wwm_tokenizer.tokenize(masked_text)
      # Construct token ids and segment ids
      token_ids = torch.tensor([self.bert_wwm_tokenizer.convert_tokens_to_ids(tokenized_masked_text)])
      segment_ids = torch.tensor([[0] * token_ids.shape[1]])
      # Set up ids on GPU
      token_ids = token_ids.to(self.device)
      segment_ids = segment_ids.to(self.device)
      # Predict masked token
      with torch.no_grad():
        outputs = self.bert_wwm_model(token_ids, token_type_ids = segment_ids)
        scores = outputs[0][0,i+1]
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
  def correctSpellingError(self, text: str, err_positions: Set[int], predictions, ne_positions: Set[int], candidate_num: int) -> List[str]:
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
          for similar_token in self.stroke[error_token][:10]:
            starting_positions[err_position].add(similar_token)
        if(error_token in self.pinyin):
          for similar_token in self.pinyin[error_token][:10]:
            starting_positions[err_position].add(similar_token)
        for predicted_token in predictions[err_position]:
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