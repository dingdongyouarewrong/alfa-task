import os
from dotenv import load_dotenv
import logging
import constants
from LangChainWrapper import LangChainWrapper
import telebot
import threading
from LlamaCppWrapper import LlamaCppWrapper

# configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AlfaTaskTelegramBot:
    def __init__(self, telegram_bot_token, filename, llm_path, n_ctx, top_k, top_p, temperature,
                 repeat_penalty, n_parts, path_to_lora_adapter):
        self.bot = telebot.TeleBot(telegram_bot_token, parse_mode=None)
        self.llama_cpp = LlamaCppWrapper(
            model_path=llm_path,
            n_ctx=n_ctx,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            n_parts=n_parts,
            lora_path=path_to_lora_adapter
        )
        self.langchain_wrapper = LangChainWrapper(self.llama_cpp.model)
        self.langchain_wrapper.embed_file(filename)
        self.is_thinking = threading.Event()
        self.is_thinking.clear()

    def start(self):
        logging.info("The bot is ready to answer")

        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(message, constants.WELCOME_MESSAGE)

        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            question = message.text
            if self.is_thinking.is_set():
                logging.info("Bot received a message while thinking: {}".format(question))
                self.bot.reply_to(message, constants.STILL_THINKING_MESSAGE)
            else:
                logging.info("Bot received a message: {}".format(question))
                self.bot.reply_to(message, constants.START_THINKING_MESSAGE)
                threading.Thread(target=self.think_and_reply, args=(question, message)).start()

        self.bot.infinity_polling()

    def think_and_reply(self, question, message):
        try:
            self.is_thinking.set()
            logging.info("Bot is thinking...")
            response = self.langchain_wrapper.get_answer(question)
            self.bot.reply_to(message, response)
            logging.info("Bot replied: {}".format(response))
        except Exception as e:
            logging.error("An error occurred while processing the message: {}".format(e))
            self.bot.reply_to(message, constants.ERROR_MESSAGE)
        finally:
            self.is_thinking.clear()


if __name__ == '__main__':
    model_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    n_ctx = 2000
    top_k = 30
    top_p = 0.9
    temperature = 0.2
    repeat_penalty = 1.1
    n_parts = 1
    lora_path = "finetuning/ggml-adapter-model.bin"

    bot = AlfaTaskTelegramBot(os.getenv("TELEGRAM_BOT_TOKEN"), "data/Альфа-Клиент_Руководство пользователя_2017.pdf",
                              model_path, n_ctx, top_k, top_p, temperature, repeat_penalty, n_parts, lora_path)
    bot.start()
