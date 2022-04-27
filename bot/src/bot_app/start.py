from aiogram import Dispatcher
from bot_app.app import db


async def on_startup(dp: Dispatcher):
    await db.create_table()
