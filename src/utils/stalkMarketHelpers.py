"""
Misc functions for Stalk Market Predictions
Part of Stalk Market Bot.
"""

import logging
from dataclasses import dataclass

from datetime import datetime, timedelta
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
    other_data: sm.OverallPatternData
    last_pattern: int

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


    @property
    def min_max(self) -> sm.Pattern:
        return self.other_data.min_max_data


    @property
    def average(self) -> List[float]:
        return self.other_data.average_prices


    @property
    def expected_prices(self) -> List[float]:
        return self.other_data.expected_prices


    @property
    def total_probabilities(self) -> List[float]:
        return self.other_data.total_probabilities


    def pattern_probability(self, pattern_number: int) -> float:
        return self.other_data.total_probabilities[pattern_number]


class TurnipDate:
    datetime_format: str = f"%Y,%U,%w,%H"

    def __init__(self, year: Optional[int] = None, week: Optional[int] = None, day_seg: Optional[int] = None, at_timezone: str = "US/Eastern" ):

        if year is None or week is None:
            est = timezone(at_timezone)
            now = est.fromutc(datetime.utcnow())

            if day_seg is None:
                # Compute the day_segment
                day_of_week = int(now.strftime("%w"))
                if now.hour >= 12:  # Past noon.
                    day_segment = day_of_week * 2 + 1
                else:
                    day_segment = day_of_week * 2

                if day_segment == 1:  # Make sure to avoid the invalid segment 1 (Sun PM)
                    day_segment = 0
            else:
                day_segment = day_seg

            self._day_segment: int = day_segment

            # Compute the week segment
            self.week: int = int(now.strftime("%U"))  # TODO: account for beginning of the year

            # Compute the year segment
            self.year: int = now.year

        else:
            self.year: int = year
            self.week: int = week

            if day_seg == 1:  # Make sure to avoid the invalid segment 1 (Sun PM)
                raise ValueError("1 (Sun PM) is an invalid Day Segment value")

            self._day_segment: int = day_seg or 0  # default to 0 if passed day segment is None

        self.instantiated_year = self.year
        self.instantiated_week = self.week
        self.instantiated_day_seg = self.day_segment

    def __str__(self) -> str:
        return self.to_str()


    @classmethod
    def from_datetime(cls, dt: datetime, day_seg: Optional[int] = None) -> 'TurnipDate':
        if day_seg is None:
            # Compute the day_segment
            day_of_week = int(dt.strftime("%w"))
            if dt.hour >= 12:  # Past noon.
                day_segment = day_of_week * 2 + 1
            else:
                day_segment = day_of_week * 2

            if day_segment == 1:  # Make sure to avoid the invalid segment 1 (Sun PM)
                day_segment = 0
        else:
            day_segment = day_seg

        # Compute the week segment
        week: int = int(dt.strftime("%U"))  # TODO: account for beginning of the year

        # Compute the year segment
        year: int = dt.year
        return cls(year=year, week=week, day_seg=day_segment)

    @property
    def day_segment(self):
        return self._day_segment

    @day_segment.setter
    def day_segment(self, value):

        if value < 0 or value > 13:
            raise ValueError("Day Segments MUST be between 0 & 13")

        if value == 1:
            self._day_segment = 0
        else:
            self._day_segment = value

    def to_str(self, offset_time: Optional[timedelta] = None) -> str:
        dt = self.to_datetime()
        if offset_time is not None:
            dt += offset_time
        time_of_day = "Morning" if dt.hour < 12 else "Afternoon"
        return dt.strftime(f"%a {time_of_day}, %b %d")  # Sun Morning, Jan 05

    def to_week_str(self, offset_time: Optional[timedelta] = None) -> str:
        dt = self.to_datetime()
        if offset_time is not None:
            dt += offset_time
        return dt.strftime(f"%B %d")  # January 05

    def next_day(self):
        if self.day_segment == 0:
            self.day_segment = 2  # Skip 1

        elif self.day_segment == 13:  # 13 is the max day_segment. Don't increment.
            return

        else:
            self.day_segment += 1  # Increment by 1


    def prev_day(self):
        if self.day_segment == 0:  # 0 is the min day_segment. Don't decrement.
            return
        elif self.day_segment == 2:  # Skip 1
            self.day_segment = 0
        else:
            self.day_segment -= 1  # Decrement by 1


    def next_week(self, max_from_start: int = None):
        if self.week == 53:  # 53 is the max week. Don't increment.
            return
        else:
            if max_from_start is None or self.instantiated_week + max_from_start > self.week:
                self.week += 1  # Increment by 1


    def prev_week(self, max_from_start:int = None):
        if self.week == 0:  # 0 is the min week. Don't decrement.
            return
        else:
            if max_from_start is None or self.instantiated_week - max_from_start < self.week:
                self.week -= 1  # Decrement by 1


    def to_datetime(self) -> datetime:
        day_of_week = int(self.day_segment/2)
        hour_of_day = 8 if self.day_segment%2 else 12
        dt = datetime.strptime(f"{self.year},{self.week},{day_of_week},{hour_of_day}", self.datetime_format)
        return dt


async def get_prices_for_user(db_pool: 'Pool', user_id: int, date: Optional[TurnipDate] = None) -> List[db.Prices]:
    if date is None:
        date = TurnipDate()

    prices = await get_prices_for_user_on_year_week(db_pool, user_id, date.year, date.week)

    return prices


async def get_prices_for_user_on_year_week(db_pool: 'Pool', user_id: int, year: int, week: int) -> List[db.Prices]:

    week_of_year = week
    year = year

    prices = await db.get_prices(db_pool, user_id, 0, year, week_of_year)
    prices.sort(key=lambda x: x.day_segment)

    return prices


async def get_last_weeks_pattern_for_user(db_pool: 'Pool', user_id: int, date: Optional[TurnipDate] = None) -> Optional[int]:
    if date is None:
        date = TurnipDate()

    pattern = await db.get_last_pattern(db_pool, user_id, date.year, date.week)

    return pattern


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
        last_pattern = await get_last_weeks_pattern_for_user(db_pool, user.user_id)
        if last_pattern is None:
            last_pattern = -1

        if len(prices) > 0:
            predictions, other_data = sm.get_predictions(prices, last_pattern)

            d_member: 'Member' = guild.get_member(user.user_id) if guild is not None else None

            user_name = d_member.display_name if d_member is not None else f"User ID: {user.user_id}"#"Unknown"

            user_predictions.append(UserPredictions(user.user_id, user_name, predictions, other_data, last_pattern))

    # Sort the user predictions from greatest to least by using the highest Expected price.
    def sort_func(_predictions: UserPredictions):
        return max(_predictions.expected_prices)

    user_predictions.sort(reverse=True, key=sort_func)

    return user_predictions
