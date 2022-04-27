import asyncio
import asyncpg

from typing import List


class Database:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        db_url: str,
        model_names: List[str],
    ):
        self.model_names = model_names
        self.pool = loop.run_until_complete(
            asyncpg.create_pool(max_size=32, dsn=db_url)
        )

    async def create_table(self):
        async with self.pool.acquire() as con:
            await con.execute(
                """
                create table if not exists ratings(
                    id serial primary key,
                    tg_user_id varchar(64) not null,
                    model_id int not null,
                    model_name varchar(128),
                    creation_time timestamp not null default now()::timestamp)
                """
            )

    async def insert_rating(self, user_id: str, model_id: int):
        async with self.pool.acquire() as con:
            await con.execute(
                "insert into ratings values (default, $1, $2, $3, default)",
                user_id,
                model_id,
                self.model_names[model_id],
            )

    async def view_ratings(self, model_id: int) -> List[asyncpg.Record]:
        async with self.pool.acquire() as con:
            return await con.fetch(
                """
                select model_id, model_name, count(1) as amount
                from ratings where model_id = $1
                group by (model_id, model_name)
                """,
                model_id,
            )

    async def get_stats(self) -> List[asyncpg.Record]:
        async with self.pool.acquire() as con:
            return await con.fetch(
                """
                select model_id, model_name, count(1) as amount
                from ratings group by (model_id, model_name)
                """
            )

    async def close_connection(self):
        await self.pool.close()
