from aiogram import types
from aiogram.utils.emoji import emojize
from aiogram.dispatcher import FSMContext

from typing import List
from bot_app.utils.matcher.matcher import Matcher
from bot_app.utils.images.image_handler import ImageHandler
from bot_app.utils.images.image_preprocessors import (
    DefaultPreprocessor,
)
from bot_app.utils.models.models import ModelConfig, ModelData
from bot_app.utils.models.mediator import Mediator
from bot_app.states import RatingSystem
from bot_app.app import (
    client,
    dp,
    db,
    bot,
    MODEL_NAMES,
    MODELS_NUM,
    DIMS,
    MAX_ELEMENTS,
    INDEX_PATHS,
    MEDIA_PATH,
)
from bot_app.markup import inline_kb
from bot_app.message_text import (
    PRE_START_TEXT,
    START_TEXT,
    HELP_TEXT,
    SHORT_HELP,
)


image_handler = ImageHandler(
    image_preprocessors=([DefaultPreprocessor() for i in range(MODELS_NUM)])
)
matchers = [
    Matcher(DIMS[i], MAX_ELEMENTS, path_to_index=INDEX_PATHS[i])
    for i in range(MODELS_NUM)
]
model_configs = [ModelConfig(model_name) for model_name in MODEL_NAMES]


def get_most_similar_ids(image_vectors: List[ModelData]):
    nearest_vector_ids = []
    for i in range(MODELS_NUM):
        nearest_vector_id = matchers[i].get_nearest_neighbour(
            image_vectors[i].data
        )
        nearest_vector_ids.append(int(nearest_vector_id[0][0]))

    return nearest_vector_ids


@dp.message_handler(commands=["start", "help"], state="*")
async def welcome(message: types.Message):
    if message.get_command() == "/start":
        await message.answer(emojize(PRE_START_TEXT))
        await message.answer(emojize(START_TEXT))
        await message.answer(SHORT_HELP)
    else:
        await message.answer(HELP_TEXT)


@dp.message_handler(commands=["show_stats"], state="*")
async def show_stats(message: types.Message):
    stats = await db.get_stats()
    message_text = ""
    for record in stats:
        message_text += f"{record['model_id']} {record['model_name']} amount:\
            {record['amount']}\n"
    if not message_text:
        message_text = (
            "Sorry, no statistics yet, be the first to make a change!\n"
        )
    await message.answer(message_text)


@dp.message_handler(
    content_types=types.ContentType.PHOTO,
    state=[RatingSystem.processing, RatingSystem.estimating],
)
async def ignore(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state == RatingSystem.estimating.state:
        await message.answer(
            "You need to choose the most similar photo before you can proceed"
        )
    else:
        await message.answer("Last photo is still processing, please wait")


@dp.message_handler(content_types=types.ContentType.PHOTO, state="*")
async def process_photo(message: types.Message, state: FSMContext):
    await RatingSystem.processing.set()
    data = await bot.download_file_by_id(message.photo[-1].file_id)

    preprocessed_images = image_handler.preprocess_image(data)

    model_inputs = []
    for index, preprocessed_image in enumerate(preprocessed_images):
        model_inputs.append(
            ModelData(
                data=preprocessed_image, model_config=model_configs[index]
            )
        )

    mediator = Mediator(client, MODELS_NUM)
    mediator.infer(model_inputs=model_inputs)
    image_vectors = await mediator.get_results()

    ids = get_most_similar_ids(image_vectors)

    for index in ids:
        photo_path = f"{MEDIA_PATH}{index + 1}.jpg"
        await message.answer_photo(photo=types.InputFile(photo_path))

    await RatingSystem.estimating.set()
    await message.reply(
        "Choose the photo with the most similar face", reply_markup=inline_kb
    )
