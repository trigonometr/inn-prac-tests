import unittest
import time

from telethon import TelegramClient
from environs import Env

env = Env()
env.read_env()

API_ID = env.int("API_ID")
API_HASH = env.str("API_HASH")
BOT_NAME = env.str("BOT_NAME")
MODEL_NAMES = env.str("MODEL_NAMES")
MODELS_NUM = len(MODEL_NAMES)

FIXTURES_PATH = "test_bot_app/fixtures"

client = TelegramClient("anon", API_ID, API_HASH)


async def sign_in():
    await client.start()


client.loop.run_until_complete(sign_in())


class TestBot(unittest.TestCase):
    def test_bot_start(self):
        command = "/start"

        async def check_start():
            entity = await client.get_entity(BOT_NAME)
            await client.send_message(entity=entity, message=command)

            time.sleep(1)

            i = 3
            async for message in client.iter_messages(entity=entity, limit=4):
                if i == 0:
                    self.assertEqual(message.message, command)
                i -= 1

        client.loop.run_until_complete(check_start())

    def test_bot_help(self):
        command = "/help"

        async def check_help():
            entity = await client.get_entity(BOT_NAME)
            await client.send_message(entity=entity, message=command)

            time.sleep(1)

            i = 1
            async for message in client.iter_messages(entity=entity, limit=2):
                if i == 0:
                    self.assertEqual(message.message, command)
                i -= 1

        client.loop.run_until_complete(check_help())

    def test_bot_image_input(self):
        async def check_image_input():
            entity = await client.get_entity(BOT_NAME)
            await client.send_file(
                entity=entity, file=f"{FIXTURES_PATH}/face.png"
            )
            time.sleep(6)

            messages = (
                [("Choose the photo with the most similar face", False)]
                + MODELS_NUM
                * [
                    ("", True),
                ]
                + ["", True]
            )
            i = 0
            async for message in client.iter_messages(
                entity=entity, limit=(MODELS_NUM + 2)
            ):
                self.assertEqual(messages[i][0], message.message)
                self.assertEqual((message.media is not None), messages[i][1])
                i += 1

        client.loop.run_until_complete(check_image_input())

    def test_bot_show_stats(self):
        time.sleep(1)
        command = "/show_stats"

        async def check_show_stats():
            entity = await client.get_entity(BOT_NAME)
            await client.send_message(entity=entity, message=command)
            time.sleep(1)

            i = 1
            async for message in client.iter_messages(entity=entity, limit=2):
                if i == 0:
                    self.assertEqual(message.message, command)
                i -= 1

        client.loop.run_until_complete(check_show_stats())
