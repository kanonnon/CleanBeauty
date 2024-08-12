import os
import sqlite3
from datetime import datetime, date
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import FollowEvent, MessageEvent, TextMessage, TextSendMessage

# from rag import return_rag_result

load_dotenv()

line_bot_api = LineBotApi(os.environ.get('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.environ.get('LINE_CHANNEL_SECRET'))

app = Flask(__name__)

conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()


@app.route("/")
def hello_world():
    return "hello world!"


@app.route("/callback", methods=['POST'])
def callback():
    """
    Callback function for LINE webhook
    """
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(FollowEvent)
def handle_follow(event):
    """
    Handle follow event
    """
    line_id = event.source.user_id
    c.execute('INSERT INTO users (line_id,) VALUES (?,)', (line_id,))
    conn.commit()
    welcome_message = "追加ありがとうございます😆このアカウントはクリーンビューティーに関する質問について、論文を基にお答えします。"
    line_bot_api.push_message(
        to=line_id,
        messages=[
            TextSendMessage(text=welcome_message)
        ]
    )


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """
    Handle text message
    """
    line_id = event.source.user_id
    question = event.message.text
    
    c.execute('SELECT id FROM users WHERE line_id = ?', (line_id,))
    user = c.fetchone()
    user_id = user[0]

    c.execute('INSERT INTO questions (user_id, question) VALUES (?,?)', (user_id, question))
    conn.commit()
    result = return_rag_result(question)
    line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=result)
            ]
    )
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

