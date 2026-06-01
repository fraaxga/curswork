# Чат-бот на питоне для помощи в работе с документов ПОПАТКУС НИУ ВШЭ

pip install -r requirements.txt

проект разделен на две части
1. retriever - фрагментация документа
2. generator - генерация ответа на основе найденных фрагментов с помощью LLM

поднять бота: PYTHONPATH=. python scripts/run_bot.py
