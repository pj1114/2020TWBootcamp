from typing import *
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from .trie import Trie
import codecs
import pickle

# Utility functions
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
		return trie

	# Load token-level similar stroke dictionary
	def loadStroke(self, path: str) -> Dict[str,List[str]]:
		with open(path, "rb") as stroke_file:
			stroke_dict = pickle.load(stroke_file)
		return stroke_dict

	# Load token-level similar pinyin dictionary
	def loadPinYin(self, path: str) -> Dict[str,List[str]]:
		with open(path, "rb") as pinyin_file:
			pinyin_dict = pickle.load(pinyin_file)
		return pinyin_dict

	# Load common character set
	def loadCharSet(self, path: str) -> set:
		chars_set = set()
		with open(path, 'r', encoding='utf-8') as char_set_file:
			for char in char_set_file:
				chars_set.add(char.replace('\n', ''))
		return chars_set

	# Load custom confusion dict
	def loadCustomConfusion(self, path: str) -> Dict[str,str]:
		custom_confusion_dict = dict()
		with open(path, "r") as file:
			for line in file:
				line = line.replace('\n', '').split('\t')
				custom_confusion_dict[line[0]] = line[1]
		return custom_confusion_dict

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

	# Check if a character is a chinese character
	def isChineseChar(self, character: str) -> bool:
		if('\u4e00' <= character <= '\u9fff'):
			return True
		else:
			return False

# Utility variables
spelling_error_detection_reply_template = {
  "type": "bubble",
  "size": "kilo",
  "body": {
    "type": "box",
    "layout": "vertical",
    "contents": [
      {
        "type": "text",
        "text": "文本偵錯結果",
        "weight": "bold",
        "style": "normal",
        "size": "md",
        "color": "#1DB446"
      },
      {
        "type": "text",
        "text": "Spelling Error Detection Result",
        "weight": "bold",
        "style": "normal",
        "size": "sm",
        "margin": "sm",
        "color": "#aaaaaa"
      },
      {
        "type": "separator",
        "margin": "lg"
      },
      {
        "type": "text",
        "text": "輸入",
        "weight": "bold",
        "style": "normal",
        "size": "sm",
        "margin": "lg",
        "color": "#5299CB"
      },
      {
        "type": "text",
        "text": "Input",
        "weight": "bold",
        "style": "normal",
        "size": "sm",
        "margin": "sm",
        "color": "#aaaaaa"
      },
      {
        "type": "text",
        "text": "placeholder",
        "weight": "regular",
        "style": "normal",
        "size": "sm",
        "margin": "sm",
        "wrap": True
      },
      {
        "type": "separator",
        "margin": "lg"
      },
      {
        "type": "text",
        "text": "輸出",
        "weight": "bold",
        "style": "normal",
        "size": "sm",
        "margin": "lg",
        "color": "#5299CB"
      },
      {
        "type": "text",
        "text": "Output",
        "weight": "bold",
        "style": "normal",
        "size": "sm",
        "margin": "sm",
        "color": "#aaaaaa"
      },
      {
        "type": "text",
        "contents": [
          {
            "type": "span",
            "text": "placeholder",
            "size": "sm"
          }
        ],
        "margin": "sm",
        "wrap": True
      }
    ]
  }
}
spelling_error_detection_output_span_template = {
  "type": "span",
  "text": "placeholder",
  "size": "sm"
}
spelling_error_detection_output_error_span_template = {
  "type": "span",
  "text": "placeholder",
  "size": "sm",
  "color": "#CD4F39"
}
carousel_menu = {
  "type": "carousel",
  "contents": [
    {
      "type": "bubble",
      "size": "kilo",
      "header": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "文本偵錯",
            "color": "#ffffff",
            "align": "start",
            "size": "xl",
            "gravity": "center",
            "decoration": "none",
            "weight": "bold",
            "wrap": True
          },
          {
            "type": "text",
            "text": "Spelling Error Detection",
            "color": "#ffffff",
            "align": "start",
            "size": "md",
            "gravity": "center",
            "margin": "lg",
            "weight": "regular",
            "wrap": True
          }
        ],
        "backgroundColor": "#27ACB2",
        "paddingTop": "19px",
        "paddingAll": "12px",
        "paddingBottom": "25px",
        "margin": "none"
      },
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "text",
                "text": "Introduction: ",
                "color": "#8C8C8C",
                "size": "sm",
                "wrap": True
              },
              {
                "type": "text",
                "text": "幫助使用者於文本中自動偵測錯字，節省使用者大量的時間。",
                "wrap": True,
                "margin": "lg"
              },
              {
                "type": "text",
                "text": "A spelling error detection machine for users intending to look for typos in their articles.",
                "wrap": True,
                "margin": "lg"
              }
            ],
            "flex": 1,
            "height": "230px"
          },
          {
            "type": "box",
            "layout": "baseline",
            "contents": [
              {
                "type": "text",
                "text": "Detect",
                "action": {
                  "type": "message",
                  "label": "Detect",
                  "text": "Detect"
                },
                "color": "#1093d4",
                "position": "relative",
                "decoration": "none",
                "style": "normal",
                "weight": "regular",
                "align": "center",
                "offsetTop": "3px"
              }
            ],
            "borderColor": "#1093d4",
            "borderWidth": "1px",
            "cornerRadius": "5px",
            "height": "30px"
          }
        ],
        "spacing": "md",
        "paddingAll": "12px",
        "height": "300px"
      },
      "styles": {
        "footer": {
          "separator": False
        }
      }
    },
    {
      "type": "bubble",
      "size": "kilo",
      "header": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "文本修正",
            "color": "#ffffff",
            "align": "start",
            "size": "xl",
            "gravity": "center",
            "decoration": "none",
            "weight": "bold"
          },
          {
            "type": "text",
            "text": "Spelling Error Correction",
            "color": "#ffffff",
            "align": "start",
            "size": "md",
            "gravity": "center",
            "margin": "lg",
            "weight": "regular",
            "wrap": True
          }
        ],
        "backgroundColor": "#FF6B6E",
        "paddingTop": "19px",
        "paddingAll": "12px",
        "paddingBottom": "25px",
        "margin": "none"
      },
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "text",
                "text": "Introduction: ",
                "color": "#8C8C8C",
                "size": "sm",
                "wrap": True
              },
              {
                "type": "text",
                "wrap": True,
                "text": "幫助使用者自動改正錯字",
                "margin": "sm",
                "size": "xs"
              },
              {
                "type": "text",
                "text": "A spelling error correction machine for users intending to correct typos automatically in their articles.",
                "wrap": True,
                "size": "xs"
              },
              {
                "type": "separator",
                "margin": "sm"
              },
              {
                "type": "text",
                "text": "✔️ Correct + Show Typo: 改正錯字並顯示錯字的位置",
                "wrap": True,
                "size": "xxs",
                "color": "#06293A",
                "margin": "sm"
              },
              {
                "type": "text",
                "text": "✔️ Correct Only: 只自動改正錯字",
                "color": "#06293A",
                "wrap": True,
                "size": "xxs"
              }
            ],
            "flex": 1,
            "height": "188px",
            "margin": "xs"
          },
          {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "box",
                "layout": "baseline",
                "contents": [
                  {
                    "type": "text",
                    "text": "Correct + Show Typo",
                    "action": {
                      "type": "message",
                      "label": "Correct and Show Typo",
                      "text": "Correct and Show Typo"
                    },
                    "color": "#1093d4",
                    "align": "center",
                    "style": "normal",
                    "offsetTop": "3px"
                  }
                ],
                "borderColor": "#1093d4",
                "borderWidth": "1px",
                "cornerRadius": "5px",
                "height": "30px",
                "offsetTop": "3px"
              },
              {
                "type": "box",
                "layout": "baseline",
                "contents": [
                  {
                    "type": "text",
                    "text": "Correct Only",
                    "action": {
                      "type": "message",
                      "label": "Correct Only",
                      "text": "Correct Only"
                    },
                    "color": "#1093d4",
                    "align": "center",
                    "weight": "regular",
                    "style": "normal",
                    "offsetTop": "3px"
                  }
                ],
                "height": "30px",
                "borderWidth": "1px",
                "borderColor": "#1093d4",
                "cornerRadius": "5px",
                "offsetTop": "13px"
              }
            ],
            "height": "90px"
          }
        ],
        "spacing": "md",
        "paddingAll": "12px",
        "height": "300px"
      },
      "styles": {
        "footer": {
          "separator": False
        }
      }
    },
    {
      "type": "bubble",
      "size": "kilo",
      "header": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "文法小老師",
            "color": "#ffffff",
            "align": "start",
            "size": "xl",
            "gravity": "center",
            "decoration": "none",
            "weight": "bold"
          },
          {
            "type": "text",
            "text": "Grammar Tutor",
            "color": "#ffffff",
            "align": "start",
            "size": "md",
            "gravity": "center",
            "margin": "lg",
            "weight": "regular",
            "wrap": True
          }
        ],
        "backgroundColor": "#A17DF5",
        "paddingTop": "19px",
        "paddingAll": "12px",
        "paddingBottom": "25px",
        "margin": "none"
      },
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "text",
                "text": "Introduction:",
                "color": "#8C8C8C",
                "size": "sm",
                "wrap": True
              },
              {
                "type": "text",
                "text": "幫助中文學習者改正短句的錯別字及文法",
                "wrap": True,
                "margin": "lg"
              },
              {
                "type": "text",
                "text": "A tool for Chinese learners to correct grammatical errors and typos, and learn the proper usage",
                "wrap": True,
                "margin": "lg"
              }
            ],
            "flex": 1,
            "height": "230px"
          },
          {
            "type": "box",
            "layout": "baseline",
            "contents": [
              {
                "type": "text",
                "text": "Start",
                "action": {
                  "type": "message",
                  "label": "Start",
                  "text": "Start"
                },
                "color": "#1093d4",
                "position": "relative",
                "decoration": "none",
                "style": "normal",
                "weight": "regular",
                "align": "center",
                "offsetTop": "3px"
              }
            ],
            "borderColor": "#1093d4",
            "borderWidth": "1px",
            "cornerRadius": "5px",
            "height": "30px"
          }
        ],
        "spacing": "md",
        "paddingAll": "12px",
        "height": "300px"
      },
      "styles": {
        "footer": {
          "separator": False
        }
      }
    },
    {
      "type": "bubble",
      "size": "kilo",
      "header": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "text",
            "text": "近似詞推薦",
            "color": "#ffffff",
            "align": "start",
            "size": "xl",
            "gravity": "center",
            "decoration": "none",
            "weight": "bold"
          },
          {
            "type": "text",
            "text": "Synonym Guidance",
            "color": "#ffffff",
            "align": "start",
            "size": "md",
            "gravity": "center",
            "margin": "lg",
            "weight": "regular",
            "wrap": True
          }
        ],
        "backgroundColor": "#0367D3",
        "paddingTop": "19px",
        "paddingAll": "12px",
        "paddingBottom": "25px",
        "margin": "none"
      },
      "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
          {
            "type": "box",
            "layout": "vertical",
            "contents": [
              {
                "type": "text",
                "text": "Introduction: ",
                "color": "#8C8C8C",
                "size": "sm",
                "wrap": True
              },
              {
                "type": "text",
                "text": "幫助使用者尋找句子中詞語的相關替換字，添增文本的色彩",
                "wrap": True,
                "margin": "lg"
              },
              {
                "type": "text",
                "text": "A tool for users to look for synonyms in their articles.",
                "margin": "lg",
                "wrap": True
              }
            ],
            "flex": 1,
            "height": "230px"
          },
          {
            "type": "box",
            "layout": "baseline",
            "contents": [
              {
                "type": "text",
                "text": "Recommend",
                "action": {
                  "type": "message",
                  "label": "Recommend",
                  "text": "Recommend"
                },
                "color": "#1093d4",
                "position": "relative",
                "decoration": "none",
                "style": "normal",
                "weight": "regular",
                "align": "center",
                "offsetTop": "3px"
              }
            ],
            "borderColor": "#1093d4",
            "borderWidth": "1px",
            "cornerRadius": "5px",
            "height": "30px"
          }
        ],
        "spacing": "md",
        "paddingAll": "12px",
        "height": "300px"
      },
      "styles": {
        "footer": {
          "separator": False
        }
      }
    }
  ]
}