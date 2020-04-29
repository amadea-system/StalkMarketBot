"""
For use with StalkMarketBot

"""

import math
import time
import json
import logging
import functools
import statistics as stats
from typing import List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import asyncpg
from discord import Invite, Message


class DBPerformance:

    def __init__(self):
        self.time = defaultdict(list)

    def avg(self, key: str):
        return stats.mean(self.time[key])

    def all_avg(self):
        avgs = {}
        for key, value in self.time.items():
            avgs[key] = stats.mean(value)
        return avgs

    def stats(self):
        statistics = {}
        for key, value in self.time.items():
            loop_stats = {}
            try:
                loop_stats['avg'] = stats.mean(value)
            except stats.StatisticsError:
                loop_stats['avg'] = -1

            try:
                loop_stats['med'] = stats.median(value)
            except stats.StatisticsError:
                loop_stats['med'] = -1

            try:
                loop_stats['max'] = max(value)
            except stats.StatisticsError:
                loop_stats['max'] = -1

            try:
                loop_stats['min'] = min(value)
            except stats.StatisticsError:
                loop_stats['min'] = -1

            loop_stats['calls'] = len(value)

            statistics[key] = loop_stats
        return statistics


db_perf = DBPerformance()


async def create_db_pool(uri: str) -> asyncpg.pool.Pool:

    # FIXME: Error Handling
    async def init_connection(conn):
        await conn.set_type_codec('json',
                                  encoder=json.dumps,
                                  decoder=json.loads,
                                  schema='pg_catalog')

    pool: asyncpg.pool.Pool = await asyncpg.create_pool(uri, init=init_connection)

    return pool


def db_deco(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            response = await func(*args, **kwargs)
            end_time = time.perf_counter()
            db_perf.time[func.__name__].append((end_time - start_time) * 1000)

            if len(args) > 1:
                logging.info("DB Query {} from {} in {:.3f} ms.".format(func.__name__, args[1], (end_time - start_time) * 1000))
            else:
                logging.info("DB Query {} in {:.3f} ms.".format(func.__name__, (end_time - start_time) * 1000))
            return response
        except asyncpg.exceptions.PostgresError:
            logging.exception("Error attempting database query: {} for server: {}".format(func.__name__, args[1]))
    return wrapper


@dataclass
class Prices:
    user_id: int
    account_id: int
    year: int
    week: int
    day_segment: int
    price: int


@dataclass
class User:
    server_id: int
    user_id: int
    account_id: int
    account_name: str


@db_deco
async def does_account_exist(pool, sid: int, user_id: int):
    async with pool.acquire() as conn:
        response = await conn.fetchval("select exists(select 1 from accounts where server_id = $1 AND user_id = $2)", sid, user_id)
        return response


@db_deco
async def add_account(pool, sid: int, user_id: int, account_id: int, account_name: str):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO accounts(server_id, user_id, account_id, account_name) VALUES($1, $2, $3, $4)
            ON CONFLICT (server_id, user_id, account_id)
            DO NOTHING
            """,
            sid, user_id, account_id, account_name)


@db_deco
async def remove_account(pool, sid: int, user_id: int, account_id: int):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM accounts WHERE server_id = $1 AND user_id = $2 AND account_id = $3", sid, user_id, account_id)


@db_deco
async def get_accounts(pool, sid: int, user_id: int) -> List[User]:
    async with pool.acquire() as conn:
        raw_rows = await conn.fetch('SELECT * FROM accounts WHERE server_id = $1 AND user_id = $2', sid, user_id)
        return [User(**row) for row in raw_rows]


@db_deco
async def get_all_accounts_for_guild(pool, sid: int) -> List[User]:
    async with pool.acquire() as conn:
        raw_rows = await conn.fetch('SELECT * FROM accounts WHERE server_id = $1', sid)
        return [User(**row) for row in raw_rows]


@db_deco
async def get_prices(pool, user_id: int, account_id: int, year: int, week: int) -> List[Prices]:
    async with pool.acquire() as conn:
        raw_rows = await conn.fetch('SELECT * FROM prices WHERE user_id = $1 AND account_id = $2 AND year = $3 and week = $4', user_id, account_id, year, week)
        return [Prices(**row) for row in raw_rows]


@db_deco
async def add_price(pool, price: Prices):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO prices(user_id, account_id, year, week, day_segment, price) VALUES($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, account_id, year, week, day_segment)
            DO UPDATE
            SET price = EXCLUDED.price
            """,
            price.user_id, price.account_id, price.year, price.week, price.day_segment, price.price)


@db_deco
async def update_price(pool, price: Prices):
    async with pool.acquire() as conn:
        await conn.execute("UPDATE prices SET price = $1 WHERE user_id =$2 AND account_id = $3 AND year = $4 AND week = $5 AND day_segment = $6",
                           price.price, price.user_id, price.account_id, price.year, price.week, price.day_segment)


@db_deco
async def remove_price(pool, price: Prices):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM prices WHERE user_id = $1 AND account_id = $2 AND year = $3 AND week = $4 AND day_segment = $5",
                           price.user_id, price.account_id, price.year, price.week, price.day_segment)


# --- Week Info Functions --- #

async def set_this_weeks_pattern(pool, user_id: int, year: int, week: int, pattern_num: Optional[int]):
    """This sets the current weeks pattern for the date given."""
    # pattern_num = pattern_num or -1  # -1 == Not yet set
    week = week + 1  # FIXME: Account for year rollover.
    await set_pattern(pool, user_id, year, week, pattern_num)


@db_deco
async def set_pattern(pool, user_id: int, year: int, week: int, pattern_num: Optional[int]):
    """This sets last weeks pattern OF THE date given."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO week_info(user_id, account_id, year, week, previous_pattern) VALUES($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, account_id, year, week)
            DO UPDATE
            SET previous_pattern = EXCLUDED.previous_pattern
            """, user_id, 0, year, week, pattern_num)


@db_deco
async def get_last_pattern(pool, user_id: int, year: int, week: int) -> Optional[int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM week_info WHERE user_id = $1 AND account_id = $2 AND year = $3 AND week = $4", user_id, 0, year, week)
        if row is None:
            return None
        else:
            return row['previous_pattern']


@db_deco
async def set_first_time_buyer(pool, user_id: int, year: int, week: int, first_time_buyer: bool):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO week_info(user_id, account_id, year, week, first_time_buyer) VALUES($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, account_id, year, week)
            DO UPDATE
            SET first_time_buyer = EXCLUDED.first_time_buyer
            """, user_id, 0, year, week, first_time_buyer)


@db_deco
async def get_first_time_buyer(pool, user_id: int, year: int, week: int) -> Optional[bool]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM week_info WHERE user_id = $1 AND account_id = $2 AND year = $3 AND week = $4", user_id, 0,
            year, week)
        if row is None:
            return None
        else:
            return row['first_time_buyer']


async def create_tables(pool):
    # Create tables

    async with pool.acquire() as conn:

        await conn.execute('''
                           CREATE TABLE if not exists accounts(
                               server_id       BIGINT,
                               user_id         BIGINT,
                               account_id      BIGINT DEFAULT 0,
                               account_name    TEXT, 
                               PRIMARY KEY (server_id, user_id, account_id)
                           )
                       ''')

        await conn.execute('''
                           CREATE TABLE if not exists prices(
                               user_id         BIGINT,
                               account_id      BIGINT DEFAULT 0,
                               year            INTEGER,
                               week            INTEGER,
                               day_segment     INTEGER,
                               price           INTEGER,
                               PRIMARY KEY (user_id, account_id, year, week, day_segment)
                               )
                           ''')

        # previous_pattern is LAST weeks pattern as that is what is relevent to THIS week.
        await conn.execute('''
                                   CREATE TABLE if not exists week_info(
                                       user_id             BIGINT,
                                       account_id          BIGINT DEFAULT 0,
                                       year                INTEGER,
                                       week                INTEGER,
                                       first_time_buyer    BOOL DEFAULT FALSE,
                                       previous_pattern    INTEGER DEFAULT null, 
                                       PRIMARY KEY (user_id, account_id, year, week)
                                       )
                                   ''')


