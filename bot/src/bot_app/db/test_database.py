from numpy import insert
import pytest
import psycopg2
import asyncio

from environs import Env
from database import Database

env = Env()
env.read_env()

TEST_DB_URL = env.str("TEST_DB_URL")

model_names = [f"model{i}" for i in range(10)]

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@pytest.fixture
def db():
    db = Database(loop, TEST_DB_URL, model_names)
    loop.run_until_complete(db.create_table())

    yield db

    loop.run_until_complete(db.close_connection())

    conn = psycopg2.connect(TEST_DB_URL)
    with conn:
        with conn.cursor() as curs:
            curs.execute("drop table ratings")


def test_create_table(db):
    pass


@pytest.mark.parametrize(
    "values",
    [
        [("318591385", 1), ("1385719835", 5)],
        [("3918569513", 2), ("9358313aF", 2), ("487155X", 3)],
        [("TG_ID", 1) for i in range(64)],
        [("TG_ID", j) for i in range(8) for j in range(8)],
    ],
)
def test_insert_ratings(db, values):
    async def insert_ratings():
        for value in values:
            await db.insert_rating(*value)

    loop.run_until_complete(insert_ratings())

    conn = psycopg2.connect(TEST_DB_URL)
    with conn:
        with conn.cursor() as curs:
            curs.execute("select * from ratings")
            content = curs.fetchall()
            for value, content_el in zip(values, content):
                assert value == content_el[1:3]


@pytest.mark.parametrize(
    "values,to_view,expected",
    [
        ([("318591385", 1), ("1385719835", 5)], 5, 1),
        ([("3918569513", 2), ("9358313aF", 2), ("487155X", 3)], 2, 2),
        ([("TG_ID", 1) for i in range(64)], 1, 64),
        ([("TG_ID", j) for i in range(8) for j in range(8)], 4, 8),
    ],
)
def test_view_ratings(db, values, to_view, expected):
    results = None

    async def get_view():
        for value in values:
            await db.insert_rating(*value)
        nonlocal results
        results = await db.view_ratings(to_view)

    loop.run_until_complete(get_view())

    for result in results:
        assert result["model_id"] == to_view
        assert result["model_name"] == model_names[to_view]
        assert result["amount"] == expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([("318591385", 1), ("1385719835", 5)], {1: 1, 5: 1}),
        ([("3918569513", 2), ("9358313aF", 2), ("487155X", 3)], {2: 2, 3: 1}),
        ([("TG_ID", 1) for i in range(64)], {1: 64}),
        (
            [("TG_ID", j) for i in range(8) for j in range(8)],
            {i: 8 for i in range(8)},
        ),
    ],
)
def test_get_stats(db, values, expected):
    results = None

    async def get_view():
        for value in values:
            await db.insert_rating(*value)
        nonlocal results
        results = await db.get_stats()

    loop.run_until_complete(get_view())

    for result in results:
        assert expected[result["model_id"]] == result["amount"]
