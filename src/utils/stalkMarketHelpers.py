"""
Misc functions for Stalk Market Predictions
Part of Stalk Market Bot.
"""

import logging
from dataclasses import dataclass

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple, NamedTuple, Type, Any

# from timezonefinder import TimezoneFinder
from pytz import timezone
# import pytz

import db

# from cogs.stalkMarket import UserPredictions
from utils import stalkMarketPredictions as sm

from utils.misc import async_perf_timer


if TYPE_CHECKING:
    from asyncpg.pool import Pool
    from discord import Guild, Member

log = logging.getLogger(__name__)

# --- ClaSSES --- #

@dataclass
class UserPredictions:
    user_id: int
    user_name: str
    patterns: List[sm.Pattern]
    min_max: sm.Pattern
    average: List[float]

    def best(self) -> Optional[sm.Pattern]:
        return self.patterns[0] if len(self.patterns) > 0 else None

    def best_type(self):
        pass

    def prediction_count(self) -> Optional[Tuple[int, int, int, int]]:
        if len(self.patterns) == 0:
            return None
        count = (
            len(list(filter(lambda x: x.number == 0, self.patterns))),
            len(list(filter(lambda x: x.number == 1, self.patterns))),
            len(list(filter(lambda x: x.number == 2, self.patterns))),
            len(list(filter(lambda x: x.number == 3, self.patterns))),
        )
        return count


async def get_prices_for_user(db_pool: 'Pool', user_id: int, date=None) -> List[db.Prices]:
    if date is None:
        est = timezone('US/Eastern')
        date = est.fromutc(datetime.utcnow())

    week_of_year = int(date.strftime("%U"))  # TODO: account for begining of the year
    year = date.year

    prices = await get_prices_for_user_on_year_week(db_pool, user_id, year, week_of_year)

    return prices


async def get_prices_for_user_on_year_week(db_pool: 'Pool', user_id: int, year: int, week: int) -> List[db.Prices]:

    week_of_year = week
    year = year

    prices = await db.get_prices(db_pool, user_id, 0, year, week_of_year)
    prices.sort(key=lambda x: x.day_segment)

    return prices


@async_perf_timer
async def get_guild_user_predictions(db_pool: 'Pool', guild_id: Optional[int] = None, guild: Optional['Guild'] = None) -> List[UserPredictions]:
    if guild_id is None and guild is None:
        raise ValueError("Either a guild id or guild object MUST be passed.")

    if guild_id is None:
        guild_id = guild.id

    users = await db.get_all_accounts_for_guild(db_pool, guild_id)

    user_predictions = []
    for user in users:
        prices = await get_prices_for_user(db_pool, user.user_id)
        if len(prices) > 0:
            predictions, min_max, average_prices = sm.get_predictions(prices)

            d_member: 'Member' = guild.get_member(user.user_id) if guild is not None else None

            user_name = d_member.display_name if d_member is not None else f"User ID: {user.user_id}"#"Unknown"

            user_predictions.append(UserPredictions(user.user_id, user_name, predictions, min_max, average_prices))

    # Sort the user predictions from greatest to least by using the highest average price.
    def sort_func(_predictions: UserPredictions):
        return max(_predictions.average)

    user_predictions.sort(reverse=True, key=sort_func)

    return user_predictions
