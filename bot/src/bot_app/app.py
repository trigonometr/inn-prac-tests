import asyncio
import tritonclient.http as httpclient

from gevent import monkey
from environs import Env
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from bot_app.db.database import Database

monkey.patch_all()

env = Env()
env.read_env()

BOT_TOKEN = env.str("BOT_TOKEN")

MODEL_NAMES = env.list("MODEL_NAMES")
MODELS_NUM = len(MODEL_NAMES)
DIMS = env.list("DIMS", subcast=int)
CONNECTIONS_NUM = env.int("CONNECTIONS_NUM")
URL = env.str("URL")

MAX_ELEMENTS = env.int("MAX_ELEMENTS")
INDEX_PATHS = env.list("INDEX_PATHS")

MEDIA_PATH = env.str("MEDIA_PATH")

TESTING = env.bool("TESTING")

if TESTING:
    DB_URL = env.str("TEST_DB_URL")
else:
    DB_URL = env.str("DATABASE_URL")


bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=MemoryStorage())


loop = asyncio.get_event_loop()
db = Database(loop, DB_URL, MODEL_NAMES)

client = httpclient.InferenceServerClient(url=URL, concurrency=CONNECTIONS_NUM)
