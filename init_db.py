import asyncio

from config.database import Base, async_engine
from models.knowledge_file import KnowledgeFile


async def create_tables():
    async with async_engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all
        )

    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_tables())

    print("数据库表创建成功")