import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import FollowEvent, MessageEvent, TextMessage, TextSendMessage
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from rag import return_rag_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

app = Flask(__name__)

cred = credentials.Certificate("/etc/secrets/firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://cbj-scholar-default-rtdb.firebaseio.com',
    'databaseAuthVariableOverride': {
        'uid': 'my-service-worker'
    }
})


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
    line_user_name = line_bot_api.get_profile(line_id).display_name
    users_ref = db.reference('/users')
    users_ref.push({
        'line_id': line_id,
        'line_user_name': line_user_name
    })
    welcome_message = f"{line_user_name}さん、追加ありがとうございます🫶\n\nこのアカウントはクリーンビューティーに関する質問について、論文の情報をもとに丁寧にお答えします。1問1答形式でお答えしますので、前の会話を考慮できないことに注意してください。"
    line_bot_api.push_message(
        to=line_id,
        messages=[
            TextSendMessage(text=welcome_message)
        ]
    )


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_id = event.source.user_id
    question = event.message.text

    users_ref = db.reference('/users')
    try:
        user = users_ref.order_by_child('line_id').equal_to(line_id).get()
        if not user:
            logger.warning(f"User information not found: {line_id}")
            error_message = "ユーザー情報が見つかりません。最初に友達登録をしてください。"
            line_bot_api.reply_message(
                event.reply_token,
                [
                    TextSendMessage(text=error_message)
                ]
            )
            return

        user_id = list(user.keys())[0]
        logger.info(f"Successfully retrieved user ID: {user_id}")

    except Exception as e:
        logger.error(f"Error retrieving user information from Firebase: {e}")
        error_message = "ユーザー情報の取得に失敗しました。もう一度お試しください。"
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=error_message)
            ]
        )
        return

    try:
        questions_ref = db.reference('/questions')
        questions_ref.push({
            'user_id': user_id,
            'question': question,
            'date': datetime.now().isoformat()
        })
        logger.info(f"Question added to Firebase: {question}")
        
    except Exception as e:
        logger.error(f"Error adding question to Firebase: {e}")
        error_message = "質問の記録に失敗しました。もう一度お試しください。"
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=error_message)
            ]
        )
        return

    try:
        result = return_rag_result(question)
        if not result:
            logger.warning("Received empty response from RAG. Using default message.")
            result = "申し訳ありませんが、適切な回答を見つけることができませんでした。"

        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=result)
            ]
        )
        logger.info("Response sent to the user.")

    except Exception as e:
        logger.error(f"Error retrieving RAG result: {e}")
        error_message = "回答を取得する際にエラーが発生しました。もう一度お試しください。"
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=error_message)
            ]
        )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
