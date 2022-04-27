from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from bot_app.app import MODELS_NUM

buttons_text = [str(i + 1) for i in range(MODELS_NUM)]
inline_kb = InlineKeyboardMarkup()

for button_text in buttons_text:
    inline_kb.add(InlineKeyboardButton(button_text, callback_data=button_text))
