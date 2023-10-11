PROMPT_TEMPLATE_STRING = '''Контекст: {context}

Основываясь на контексте выше, ответь на вопрос:
Вопрос: {question}
'''
EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-base"
WELCOME_MESSAGE = "Это бот для тестового задания Альфа-Банк Беларусь, автор - Дмитрий Назаров. База знаний - руководство пользователя Альфа-Клиент. Бот отвечает примерно 7 минут."
STILL_THINKING_MESSAGE = "Подождите, я еще размышляю... Из-за инференса на CPU, я буду думать медленно, так что сделайте глубокий вдох, выдохните, потом ещё раз, и ещё..."
START_THINKING_MESSAGE = "Дайте мне немного времени для размышлений... Из-за инференса на CPU, я буду думать медленно, так что сделайте глубокий вдох, выдохните, потом ещё раз, и ещё..."


ERROR_MESSAGE = "Произошла ошибка, попробуйте ещё раз позже"