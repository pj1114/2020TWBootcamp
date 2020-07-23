from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import yaml
from zailagan import ZaiLaGan

# Initialize application
app = Flask(__name__)

# Load configuration file and initialize connection to line api
with open("./config.yml", "r") as config_file_yaml:
  config = yaml.load(config_file_yaml, Loader = yaml.BaseLoader)

# Channel access token and secret
line_bot_api = LineBotApi(config["Linebot"]["access_token"])
handler = WebhookHandler(config["Linebot"]["secret"]) 

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
  # Testing
  try:
    print("handling input: " + event.message.text)
    recommendations = ZLG.correctSpellingError(event.message.text, ZLG.detectSpellingError(event.message.text,1e-4), [], 5)
    response = "*****輸入*****\n" + event.message.text + "\n*****輸出*****\n"
    for i in range(len(recommendations)):
      if(i != len(recommendations)-1):
        response += str(i+1) + ". " + recommendations[i] + "\n"
      else:
        response += str(i+1) + ". " + recommendations[i]
    reply(response)
  except:
    print("failed :(")

# Run application
app.run()