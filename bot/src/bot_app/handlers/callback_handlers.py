from aiogram import types
from aiogram.dispatcher import FSMContext

from bot_app.app import dp, bot, db
from bot_app.states import RatingSystem
from bot_app.markup import buttons_text


@dp.callback_query_handler(
    lambda c: c.data in buttons_text, state=RatingSystem.estimating
)
async def process_rate(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query_id=callback_query.id)

    await db.insert_rating(
        str(callback_query.from_user.id), int(callback_query.data) - 1
    )

    await bot.send_message(callback_query.from_user.id, "Thank you!")
    await RatingSystem.start.set()
