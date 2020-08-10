from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import yaml
from zailagan import ZaiLaGan
import os
import traceback
from utilities.utils import *

# Initialize application
app = Flask(__name__)

# Load configuration file and initialize connection to line api
with open("./config.yml", "r") as config_file_yaml:
  config = yaml.load(config_file_yaml, Loader = yaml.BaseLoader)

# Channel access token and secret
line_bot_api = LineBotApi(config["Linebot"]["access_token"])
handler = WebhookHandler(config["Linebot"]["secret"]) 

# Concatenate root path with all subpaths
root_path = config["User"]["user_path"]
model_names = ["gpt2_chinese", "ner", "ngram", "w2v_model", "ws_model", "pos_model"]
data_names = ["pinyin", "stroke", "dictionary", "common_char_set", "confusion", "place", "person", "ssc", "anti_dict", "label_map"]
for model_name in model_names:
    config["Model"][model_name] = os.path.join(root_path, config["Model"][model_name])
for data_name in data_names:
    config["Data"][data_name] = os.path.join(root_path, config["Data"][data_name])

# Instantiate ZaiLaGan
ZLG = ZaiLaGan(config)
print(ZLG.wordSub_model.get_word_subs("我想要吃好吃的漢堡"))

# Handle incoming requests
@app.route("/callback", methods = ["POST"])
def callback():
  # Get X-Line-Signature header value
  signature = request.headers["X-Line-Signature"]
  # Get request body as text
  body = request.get_data(as_text = True)
  # Handle request
  try:
      handler.handle(body, signature)
  except InvalidSignatureError:
      abort(400)
  return "OK"

@handler.add(MessageEvent, message = TextMessage)
def handle_message(event):
  # Reply message to user
  def safe_reply_text(text_message):
    try:
      line_bot_api.reply_message(event.reply_token, TextSendMessage(text = text_message))
      print("Send response with reply message")
    except:
      line_bot_api.push_message(event.source.user_id, TextSendMessage(text = text_message))
      print("Send response with push message")
  def safe_reply_flex(flex_message):
    try:
      line_bot_api.reply_message(event.reply_token, flex_message)
      print("Send response with reply message")
    except:
      line_bot_api.push_message(event.source.user_id, flex_message)
      print("Send response with push message")
  try:
    # Check if the carousel menu should be returned
    if(event.message.text.lower() == "zlg"):
      handle_message.state = "init"
      safe_reply_flex(FlexSendMessage(alt_text = "功能選單", contents = carousel_menu))
      return
    # Check if the state should be changed
    if(event.message.text == "***文本偵錯***"):
      handle_message.state = "spelling_error_detection"
      safe_reply_text("Spelling error detection service activated!")
      return
    elif(event.message.text == "***文本修正***"):
      handle_message.state = "spelling_error_correction"
      safe_reply_text("Spelling error correction service activated!")
      return
    elif(event.message.text == "***文法修正***"):
      handle_message.state = "grammar_error_correction"
      safe_reply_text("Grammar error correction service activated!")
      return
    elif(event.message.text == "***近似詞推薦***"):
      handle_message.state = "synonym_recommendation"
      safe_reply_text("Synonym recommendation service activated!")
      return
    # Check the state to support different services
    if(handle_message.state == "init"):
      safe_reply_text("Please select a service from the rich menu first!")
    elif(handle_message.state == "spelling_error_detection"):
      print("Handling spelling error detection with input: " + event.message.text)
      # Perform named-entity recognition first
      ner_processed_text, ne_positions, ne_err_positions = ZLG.detectNamedEntity([event.message.text], 'detection')[0]
      ne_positions = set(ne_positions)
      # Detect spelling errors
      err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, 1e-5, 3)
      # Extract errors not included in any named-entity
      non_ne_err_positions = []
      for err_position in err_positions:
        if(err_position not in ne_positions):
          non_ne_err_positions.append(err_position)
      # Reply detection result to user
      err_positions = set(non_ne_err_positions+ne_err_positions)
      spelling_error_detection_reply = spelling_error_detection_reply_template.copy()
      # Fill in input
      spelling_error_detection_reply["body"]["contents"][5]["text"] = event.message.text
      # Fill in output
      if(len(err_positions) == 0):
        spelling_error_detection_output_span = spelling_error_detection_output_span_template.copy()
        spelling_error_detection_output_span["text"] = event.message.text
        spelling_error_detection_reply["body"]["contents"][9]["contents"] = [spelling_error_detection_output_span]
      else:
        spelling_error_detection_output_spans = []
        for i in range(len(ner_processed_text)):
          if(i in err_positions):
            spelling_error_detection_output_error_span = spelling_error_detection_output_error_span_template.copy()
            spelling_error_detection_output_error_span["text"] = ner_processed_text[i]
            spelling_error_detection_output_spans.append(spelling_error_detection_output_error_span)
          else:
            spelling_error_detection_output_span = spelling_error_detection_output_span_template.copy()
            spelling_error_detection_output_span["text"] = ner_processed_text[i]
            spelling_error_detection_output_spans.append(spelling_error_detection_output_span)
        spelling_error_detection_reply["body"]["contents"][9]["contents"] = spelling_error_detection_output_spans
      safe_reply_flex(FlexSendMessage(alt_text = "文本偵錯結果", contents = spelling_error_detection_reply))
    elif(handle_message.state == "spelling_error_correction"):
      print("Handling spelling error correction with input: " + event.message.text)
      # Correct spelling errors
      ner_processed_text, result = ZLG.divideAndCorrectSpellingError(event.message.text)
      # Reply correction result to user
      spelling_error_correction_reply = spelling_error_correction_reply_template.copy()
      # Fill in input
      spelling_error_correction_reply["body"]["contents"][5]["text"] = event.message.text
      # Fill in output
      spelling_error_correction_output_spans = []
      for i in range(len(ner_processed_text)):
        if(ner_processed_text[i] == result[i]):
          spelling_error_correction_output_span = spelling_error_correction_output_span_template.copy()
          spelling_error_correction_output_span["text"] = ner_processed_text[i]
          spelling_error_correction_output_spans.append(spelling_error_correction_output_span)
        else:
          spelling_error_correction_output_typo_span = spelling_error_correction_output_typo_span_template.copy()
          spelling_error_correction_output_typo_span["text"] = ner_processed_text[i]
          spelling_error_correction_output_spans.append(spelling_error_correction_output_typo_span)
          spelling_error_correction_output_correction_span = spelling_error_correction_output_correction_span_template.copy()
          spelling_error_correction_output_correction_span["text"] = result[i]
          spelling_error_correction_output_spans.append(spelling_error_correction_output_correction_span)
      spelling_error_correction_reply["body"]["contents"][9]["contents"] = spelling_error_correction_output_spans
      safe_reply_flex(FlexSendMessage(alt_text = "文本修正結果", contents = spelling_error_correction_reply))
    elif(handle_message.state == "grammar_error_correction"):
      print("Handling grammar error correction with input: " + event.message.text)
      safe_reply_text("Sorry, grammar error correction service can't be supported now...")
    else:
      print("Handling synonym recommendation with input: " + event.message.text)
      safe_reply_text("Sorry, synonym recommendation service can't be supported now...")
  except:
    traceback.print_exc()

# Initialize state of function
setattr(handle_message, "state", "init")

# Run application
app.run()