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
model_names = ["gpt2_chinese", "ner"]
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
  # Correct input sentence's spelling errors
  try:
    print("handling input: " + event.message.text)
    # Perform named-entity recognition first
    ner_processed_text, ne_positions = ZLG.detectNamedEntity([event.message.text])[0]
    ne_positions = set(ne_positions)
    # Call different correctors according to length of input
    if(len(event.message.text) <= 8):
      print(ZLG.bertDetectAndCorrect(ner_processed_text, 3))
    else:
      # Detect spelling errors
      err_positions, bert_predictions = ZLG.detectSpellingError(ner_processed_text, 1e-2, 3)
      err_positions = set(err_positions)
      # Count the number of errors that are not included in any named-entity
      non_ne_err_count = 0
      for err_position in err_positions:
        if(err_position not in ne_positions):
          non_ne_err_count += 1
      # Too many detected errors
      if(non_ne_err_count >= 4):
        reply("系統偵測到的錯字過多，很抱歉我們無法幫助您 :(")
      # Correct spelling errors
      else:
        recommendations = ZLG.correctSpellingError(ner_processed_text, err_positions, bert_predictions, ne_positions, 5, 1.5)
        response = "*****輸入*****\n" + event.message.text + "\n*****輸出*****\n"
        for i in range(len(recommendations)):
          if(i != len(recommendations)-1):
            response += str(i+1) + ". " + recommendations[i][0] + " (" + str(recommendations[i][2]) + ")" + "\n"
          else:
            response += str(i+1) + ". " + recommendations[i][0] + " (" + str(recommendations[i][2]) + ")"
        reply(response)
  except:
    print("failed :(")
    traceback.print_exc()

# Run application
app.run()