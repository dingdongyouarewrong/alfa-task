# alfa-task

Для запуска необходимо:
1. Скачать LLM в корень проекта по этой ссылке(https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main)
2. Добавить в корень проекта env файл c содержимым TELEGRAM_BOT_TOKEN="<ваш токен бота>"
3. Запустить файл bot.py

Для LLM finetuning нужно запустить код в ноутбуке mistral_finetune в папке finetuning, после чего скриптом convert_lora_to_ggml.py сконвертировать адаптер в ggml формат
