from aiogram import executor
from bot_app import dp
from bot_app.start import on_startup

if __name__ == '__main__':
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
