from aiogram.dispatcher.filters.state import State, StatesGroup


class RatingSystem(StatesGroup):
    start = State()
    processing = State()
    estimating = State()
