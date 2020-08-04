from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import yaml
from zailagan import ZaiLaGan
import os
import traceback

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
model_names = ["gpt2_chinese", "ner", "ngram"]
data_names = ["pinyin", "stroke", "dictionary", "common_char_set", "confusion", "place", "person", "ssc"]
for model_name in model_names:
    config["Model"][model_name] = os.path.join(root_path, config["Model"][model_name])
for data_name in data_names:
    config["Data"][data_name] = os.path.join(root_path, config["Data"][data_name])

# Instantiate ZaiLaGan
ZLG = ZaiLaGan(config)

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
  def reply(text):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = text))
  try:
    # Check if the state should be changed
    if(event.message.text == "***文本偵錯***"):
      handle_message.state = "spelling_error_detection"
      reply("Spelling error detection service activated!")
      return
    elif(event.message.text == "***文本修正***"):
      handle_message.state = "spelling_error_correction"
      reply("Spelling error correction service activated!")
      return
    elif(event.message.text == "***文法修正***"):
      handle_message.state = "grammar_error_correction"
      reply("Grammar error correction service activated!")
      return
    # Check the state to support different services
    if(handle_message.state == "init"):
      reply("Please select a service from the rich menu first!")
    elif(handle_message.state == "spelling_error_detection"):
      print("Handling spelling error detection with input: " + event.message.text)
      # Perform named-entity recognition first
      ner_processed_text, ne_positions = ZLG.detectNamedEntity([event.message.text], 'detection')[0]
      ne_positions = set(ne_positions)
      # Detect spelling errors
      err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, 1e-5, 3)
      # Extract errors not included in any named-entity
      non_ne_err_positions = []
      for err_position in err_positions:
        if(err_position not in ne_positions):
          non_ne_err_positions.append(err_position)
      # Reply detection result to user
      offset = 0
      result = event.message.text
      for position in non_ne_err_positions:
        result = result[:position+offset] + "「" + result[position+offset] + "」" + result[position+1+offset:]
        offset += 2
      reply("Spelling error detection result: \n" + result)
    elif(handle_message.state == "spelling_error_correction"):
      print("Handling spelling error correction with input: " + event.message.text)
      # Perform named-entity recognition first
      ner_processed_text, ne_positions = ZLG.detectNamedEntity([event.message.text], 'correction')[0]
      # Call different correctors according to length of input
      if(len(event.message.text) <= 8):
        result = ZLG.bertDetectAndCorrect(ner_processed_text, 3, ne_positions)[0]
        # Reply correction result to user
        reply("Spelling error correction result: \n" + result)
      else:
        ne_positions = set(ne_positions)
        # Detect spelling errors
        err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, 5e-3, 3)
        err_positions = set(err_positions)
        # Correct spelling errors
        result = ZLG.correctSpellingError(ner_processed_text, err_positions, bert_predictions, ne_positions, 1, 3)[0][0]
        # Reply correction result to user
        reply("Spelling error correction result: \n" + result)
    else:
      print("Handling grammar error correction with input: " + event.message.text)
      reply("Sorry, grammar error correction service can't be supported now...")
  except:
    traceback.print_exc()

# Initialize state of function
setattr(handle_message, "state", "init")

# Run application
app.run()