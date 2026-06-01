from telegram import Update
from telegram.ext import ContextTypes
from src.bot.rag_service import RAGService
START_TEXT = """
Привет! Я справочный бот по документу ПОПАТКУС.
Задайте интересущий Вас вопрос.
"""
HELP_TEXT = """
Я отвечаю только на вопросы, ответы на которые содержатся в документе ПОПАТКУС.
Если информации нет в документе, я дам ссылку на службу поддержки.
"""
class TelegramBot:
    def __init__(self) -> None:
        self.rag_service = RAGService()
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(START_TEXT)
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP_TEXT)
    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        user_text = update.message.text
        await update.message.reply_text("Поиск...")
        try:
            answer = self.rag_service.answer_question(user_text)
        except Exception as error:
            answer = (
                "Произошла ошибка при обработке запроса. "
            )
            print(f"Telegram bot error: {error}")
        await update.message.reply_text(answer)