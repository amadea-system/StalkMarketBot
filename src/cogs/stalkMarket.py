"""
Cog implementing discord interface of Stalk Market Predictions
Part of Stalk Market Bot.
"""

import logging
import time

from datetime import datetime, timedelta
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands
import aiohttp

from timezonefinder import TimezoneFinder
from pytz import timezone
import pytz

import dateparser
import eCommands
import db

import utils.stalkMarketPredictions as sm
from utils.stalkMarketGraphs import matplotgraph_predictions, matplotgraph_guild_predictions

from utils.stalkMarketHelpers import get_prices_for_user, get_guild_user_predictions, get_last_weeks_pattern_for_user, UserPredictions, TurnipDate, day_name_lut, am_pm_lut

from utils.uiElements import BoolPage, StringReactPage


if TYPE_CHECKING:
    from bot import GGBot

log = logging.getLogger(__name__)

day_segment_names = ["Sunday Buy Price", "N/A", "Monday Morning", "Monday Afternoon", "Tuesday Morning",
                     "Tuesday Afternoon", "Wednesday Morning", "Wednesday Afternoon", "Thursday Morning",
                     "Thursday Afternoon", "Friday Morning", "Friday Afternoon", "Saturday Morning", "Saturday Afternoon"]


# http://jkorpela.fi/chars/spaces.html
pattern_space_table = (
    "\N{HAIR SPACE}",  # 0 Roller Coaster
    "\N{SIX-PER-EM SPACE}\N{HAIR SPACE}",  # 1  Huge Spike
    "",  # 2  Always Decreasing
    "\N{SIX-PER-EM SPACE}",  # 3  Small Spike
)


number_emotes = [
    "\N{DIGIT ZERO}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT ONE}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT TWO}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT THREE}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT FOUR}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT FIVE}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT SIX}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT SEVEN}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT EIGHT}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{DIGIT NINE}\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}",
    "\N{KEYCAP TEN}",
]


tzf = TimezoneFinder()


class InvalidTimeZoneError(Exception):
    pass


async def set_time_zone(tz_name: str) -> pytz.tzinfo:
    """
    Sets the system time zone to the time zone represented by the given string.
    If `tz_name` is None or an empty string, will default to UTC.
    If `tz_name` does not represent a valid time zone string, will raise InvalidTimeZoneError.
    :raises: InvalidTimeZoneError
    :returns: The `pytz.tzinfo` instance of the newly set time zone.
    """

    try:
        tz = pytz.timezone(tz_name or "UTC")
    except pytz.UnknownTimeZoneError:
        raise InvalidTimeZoneError(tz_name)

    # await db.update_system_field(conn, tz.zone)
    return tz


async def get_timezone(ctx: commands.Context, city_query: str):

    msg = await ctx.send("\U0001F50D Searching '{}' (may take a while)...".format(city_query))

    # Look up the city on Overpass (OpenStreetMap)
    async with aiohttp.ClientSession() as sess:
        # OverpassQL is weird, but this basically searches for every node of type city with name [input].
        async with sess.get("https://nominatim.openstreetmap.org/search?city=novosibirsk&format=json&limit=1", params={"city": city_query, "format": "json", "limit": "1"}) as r:
            if r.status != 200:
                await ctx.send("\N{WARNING SIGN} OSM Nominatim API returned error. Try again.")
                return None

            data = await r.json()

    # If we didn't find a city, complain
    if not data:
        await ctx.send("\N{WARNING SIGN} City '{}' not found.".format(city_query))
        return None

    # Take the lat/long given by Overpass and put it into timezonefinder
    lat, lng = (float(data[0]["lat"]), float(data[0]["lon"]))
    timezone_name = tzf.timezone_at(lng=lng, lat=lat)

    # Also delete the original searching message
    await msg.delete()

    if not timezone_name:
        await ctx.send("\N{WARNING SIGN} Time zone for city '{}' not found. This should never happen.".format(data[0]["display_name"]))
        return None

    # This should hopefully result in a valid time zone name
    # (if not, something went wrong)

    tz = await set_time_zone(timezone_name)
    offset = tz.utcoffset(datetime.utcnow())
    offset_str = "UTC{:+02d}:{:02d}".format(int(offset.total_seconds() // 3600), int(offset.total_seconds() // 60 % 60))

    await ctx.send("Account Time zone set to {} ({}, {}).\n*Data from OpenStreetMap, queried using Nominatim.*".format(tz.tzname(datetime.utcnow()), offset_str, tz.zone))


class StalkMarket(commands.Cog):

    def __init__(self, bot: 'GGBot'):
        self.bot = bot


    @commands.guild_only()
    @eCommands.group(name="register",
                     brief="Associates your account with the current guild",
                     examples=['']
                     )
    async def register(self, ctx: commands.Context):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")
        await ctx.send(f"Your account has been associated with {ctx.guild.name}."
                       f" Group based commands such as `s;predict` will now work with your account.")


    @eCommands.group(name="add_price", aliases=["add", "ap"], brief="Add a new price at the current Eastern/US time.",
                     # description="Sets/unsets/shows the default logging channel.",  # , usage='<command> [channel]'
                     examples=['39', "42"])
    async def add_new_price_now(self, ctx: commands.Context, price: int):

        now = TurnipDate()
        await self.add_new_price_handler(ctx, price, now.year, now.week, now.day_segment)


    async def add_new_price_handler(self, ctx: commands.Context, price: int, year: int, week: int, day_segment: int):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")

        embed = discord.Embed(title="Add New Price",
                              description=f"Do you wish to set the price for {day_segment_names[day_segment]} to **{price}** Bells?")
        buttons = [
            ("✅", "accept"),
            ("\N{Leftwards Black Arrow}", "left"),
            ("\N{Black Rightwards Arrow}", "right"),
        ]

        add_prompt = StringReactPage(embed=embed, edit_in_place=True, buttons=buttons, allowable_responses=[])
        while True:
            response = await add_prompt.run(ctx)

            if response is None:
                last_embed = discord.Embed(title="❌ Price Set Canceled!",
                                           description=f"No new prices were added!")
                await add_prompt.finish(last_embed)
                return

            elif response.content() == "accept":
                last_embed = discord.Embed(title="✅ Price Set",
                                           description=f"Set the price for {day_segment_names[day_segment]} to **{price}** Bells.")
                await add_prompt.finish(last_embed)

                new_price = db.Prices(user_id=ctx.author.id, account_id=0, year=year, week=week,
                                      day_segment=day_segment, price=price)

                await db.add_price(self.bot.db_pool, new_price)
                return

            elif response.content() == "left":
                if day_segment == 0:
                    pass
                elif day_segment == 2:
                    day_segment = 0  # Skip 1
                else:
                    day_segment -= 1  # Decrement by 1

                add_prompt.embed = discord.Embed(title="Add New Price",
                                                 description=f"Do you wish to set the price for {day_segment_names[day_segment]} to **{price}** Bells?")

            elif response.content() == "right":

                if day_segment == 0:
                    day_segment = 2  # Skip 1
                elif day_segment == 13:
                    pass
                else:
                    day_segment += 1  # Increment by 1

                add_prompt.embed = discord.Embed(title="Add New Price",
                                                 description=f"Do you wish to set the price for {day_segment_names[day_segment]} to **{price}** Bells?")


    @eCommands.group(name="bulk_add", aliases=["b_a"], brief="Add a list of prices starting from Monday morning",
                     description="Add a list of prices starting from Monday morning. For prices you may not have, just type `none`.",  # , usage='<command> [channel]'
                     examples=['98 65 100 none 183 32'])
    async def bulk_add_price(self, ctx: commands.Context, *prices):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")

        if len(prices) == 0:
            await ctx.send_help(self.bulk_add_price)
            return

        if len(prices) > 12:
            await ctx.send("\N{WARNING SIGN} Too many prices entered! There are only 12 different sell prices in the week!")
            return

        parsed_prices = []
        for price in prices:
            try:
                parsed_price = int(price)
                parsed_price = parsed_price if parsed_price > 0 else None
            except ValueError:
                parsed_price = None

            parsed_prices.append(parsed_price)

        now = TurnipDate(day_seg=2)

        embed = discord.Embed(title=f"Add New Prices", description="Are you sure you want to add the following prices?")

        for price in parsed_prices:
            if price is not None:
                embed.add_field(name=day_segment_names[now.day_segment], value=f"{price} Bells")
            now.next_day() # day_segment += 1

        confirmation = BoolPage(embed=embed)

        response = await confirmation.run(ctx)
        if response:
            now.day_segment = 2
            for price in parsed_prices:
                if price is not None:
                    new_price = db.Prices(user_id=ctx.author.id, account_id=0, year=now.year, week=now.week,
                                          day_segment=now.day_segment, price=price)
                    await db.add_price(self.bot.db_pool, new_price)
                now.next_day() # day_segment += 1

            await ctx.send(f"Prices set!")
        else:
            await ctx.send(f"Canceled!")


    @eCommands.command(name="add_price_at", aliases=["add_at", "ap_at"],
                       brief="Add a new price for a given day and time period.",
                       examples=['39 Monday Morning', "120 tue am", "420 t a", "69 th p"])
    async def add_new_price_at(self, ctx: commands.Context, price: Optional[int] = None, day: Optional[str] = None,
                               time_period: Optional[str] = None):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")

        if price is None or day is None or time_period is None:  # if any arguments are omitted, send help.
            await ctx.send_help(self.add_new_price_at)
            return

        day = day.strip().lower()
        time_period = time_period.strip().lower()
        day_num = day_name_lut[day] if day in day_name_lut else None
        time_period_num = am_pm_lut[time_period] if time_period in am_pm_lut else None

        if day_num is None and time_period_num is None:
            await ctx.send(f"Error! Unable to parse the day or time period!")
            return
        elif day_num is None:
            await ctx.send(f"Error! Unable to parse the day!")
            return
        elif time_period_num is None:
            await ctx.send(f"Error! Unable to parse the time period!")
            return

        day_seg = day_num + time_period_num
        date = TurnipDate(day_seg=day_seg)

        await self.add_new_price_handler(ctx, price, date.year, date.week, date.day_segment)

    @eCommands.command(name="add_price_at_date", aliases=["add_at_date", "ap_at_d"],
                       brief="Add a new price at a specific date & time.",
                       examples=['39 1 day and 6 hours ago utc+1', "42 4/5 11:00am est"])
    async def add_new_price_at_date(self, ctx: commands.Context, price: int, *, date: str):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")

        now = dateparser.parse(date)
        if now is None:
            await ctx.send(f"Error! Unable to determine when {date} is!")
            return

        turnip_now = TurnipDate.from_datetime(now)
        await self.add_new_price_handler(ctx, price, turnip_now.year, turnip_now.week, turnip_now.day_segment)


    @eCommands.command(name="list", aliases=["list_prices"],
                       brief="Lists the current weeks recorded prices.",
                       examples=[""])
    async def list_prices(self, ctx: commands.Context, user: Optional[discord.Member] = None):

        if user is None:
            user = ctx.author

        prices = await get_prices_for_user(self.bot.db_pool, user.id)

        embed = discord.Embed(title=f"Prices for {user.display_name}")
        if len(prices) > 0:
            for price in prices:
                embed.add_field(name=day_segment_names[price.day_segment], value=f"{price.price} Bells")
        else:
            embed.description = "\N{WARNING SIGN} No prices have been recorded yet!"

        await ctx.send(embed=embed)


    @eCommands.command(name="remove",
                       brief="Allows you tto remove a recorded price from the current week",
                       examples=[''])
    async def remove(self, ctx: commands.Context):
        now = TurnipDate()

        embed = discord.Embed(title="Remove Price",
                              description=f"Do you wish to remove the recorded price for {day_segment_names[now.day_segment]}?")

        buttons = [
            ("✅", "accept"),
            ("\N{Leftwards Black Arrow}", "left"),
            ("\N{Black Rightwards Arrow}", "right"),
        ]

        remove_prompt = StringReactPage(embed=embed, edit_in_place=True, buttons=buttons, allowable_responses=[])
        while True:
            response = await remove_prompt.run(ctx)

            if response is None:
                last_embed = discord.Embed(title="❌ Price Remove Canceled!",
                                           description=f"No prices were removed!")
                await remove_prompt.finish(last_embed)
                return

            elif response.content() == "accept":
                last_embed = discord.Embed(title="✅ Price Removed",
                                           description=f"The price of for {day_segment_names[now.day_segment]} was removed.")
                await remove_prompt.finish(last_embed)

                remove_price = db.Prices(user_id=ctx.author.id, account_id=0, year=now.year, week=now.week,
                                         day_segment=now.day_segment, price=0)
                await db.remove_price(self.bot.db_pool, remove_price)

                return

            elif response.content() == "left":
                now.prev_day()

                remove_prompt.embed = discord.Embed(title="Remove Price",
                                                    description=f"Do you wish to remove the recorded price for {day_segment_names[now.day_segment]}?")

            elif response.content() == "right":
                now.next_day()

                remove_prompt.embed = discord.Embed(title="Remove Price",
                                                    description=f"Do you wish to remove the recorded price for {day_segment_names[now.day_segment]}?")


    @eCommands.group(name="set_pattern", aliases=["sp"], brief="Records what pattern you had.",
                     # description="Sets/unsets/shows the default logging channel.",  # , usage='<command> [channel]'
                     examples=[''])
    async def set_pattern(self, ctx: commands.Context):

        now = TurnipDate(day_seg=0)
        now.next_week()
        now.instantiated_week = now.week

        last_week_offset = timedelta(weeks=-1)

        button_desc = f"\n\nPress the corresponding reaction:\n" \
                      f"❓: Unknown\n" \
                      f"{number_emotes[0]}: Roller Coaster\n" \
                      f"{number_emotes[1]}: Huge Spike\n" \
                      f"{number_emotes[2]}: Always Decreasing\n" \
                      f"{number_emotes[3]}: Small Spike"

        embed = discord.Embed(title="Set Pattern",
                              description=f"Do you wish to set the pattern for the week of {now.to_week_str(last_week_offset)}?\n{button_desc}")
        buttons = [
            ("\N{Leftwards Black Arrow}", "left"),
            ("\N{Black Rightwards Arrow}", "right"),
            ("❓", "unknown"),
            (number_emotes[0], "zero"),
            (number_emotes[1], "one"),
            (number_emotes[2], "two"),
            (number_emotes[3], "three"),

        ]

        button_lut = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'unknown': -1}

        set_pattern_prompt = StringReactPage(embed=embed, edit_in_place=True, buttons=buttons, allowable_responses=[])
        while True:
            response = await set_pattern_prompt.run(ctx)

            if response is None:
                last_embed = discord.Embed(title="❌ Pattern Set Canceled!",
                                           description=f"No pattern has been set!")
                await set_pattern_prompt.finish(last_embed)
                return

            elif response.content() == "left":
                now.prev_week()

                set_pattern_prompt.embed = discord.Embed(title="Set Pattern",
                                                         description=f"Do you wish to set the pattern for the week of {now.to_week_str(last_week_offset)}?\n{button_desc}")

            elif response.content() == "right":
                now.next_week()

                set_pattern_prompt.embed = discord.Embed(title="Set Pattern",
                                                         description=f"Do you wish to set the pattern for the week of {now.to_week_str(last_week_offset)}?\n{button_desc}")

            elif response.content() in button_lut.keys():
                last_embed = discord.Embed(title="✅ Pattern Set",
                                           description=f"the pattern for the week of {now.to_week_str(last_week_offset)} has been set to {sm.pattern_definitions.name(button_lut[response.content()])}")

                await set_pattern_prompt.finish(last_embed)

                log.info(f"Setting pattern for week {now.week} to {button_lut[response.content()]}")
                await db.set_pattern(self.bot.db_pool, ctx.author.id, now.year, now.week, button_lut[response.content()])
                return


    @commands.is_owner()
    @eCommands.command(name="tpredict", #aliases=["list_prices"],
                       brief="Predicts the possible outcomes",
                       examples=[""], hidden=True)
    async def predict_prices(self, ctx: commands.Context):

        embed = discord.Embed(title=f"Price predictions for {ctx.author.display_name}")

        prices = await get_prices_for_user(self.bot.db_pool, ctx.author.id)
        # predictions, min_max = sm.get_test_predictions()
        predictions, other_data = sm.get_predictions(prices)

        desc = f"You have the following possible patterns:"

        if len(predictions) == 0:
            desc += "\n**None!!!**\n**It is likely that the dataset is incorrect.**"

        for prediction in predictions:
            # desc.append(prediction.description)

            embed.add_field(name=prediction.description,
                            value=f"```"
                                  f"Monday AM:    {prediction.prices[2]}\n"
                                  f"Monday PM:    {prediction.prices[3]}\n"
                                  f"Tuesday AM:   {prediction.prices[4]}\n"
                                  f"Tuesday PM:   {prediction.prices[5]}\n"
                                  f"Wednesday AM: {prediction.prices[6]}\n"
                                  f"Wednesday AM: {prediction.prices[7]}\n"
                                  f"Thursday AM:  {prediction.prices[8]}\n"
                                  f"Thursday AM:  {prediction.prices[9]}\n"
                                  f"Friday AM:    {prediction.prices[10]}\n"
                                  f"Friday AM:    {prediction.prices[11]}\n"
                                  f"Saturday AM:  {prediction.prices[12]}\n"
                                  f"Saturday AM:  {prediction.prices[13]}"
                                  f"```", inline=False)

        embed.description = desc

        await ctx.send(embed=embed)


    def get_spaces_for_pattern(self, pattern_num: int, patterns_seen):

        if 2 in patterns_seen:  # Always dec
            em_quads = (
                "\N{EM QUAD}\N{EM QUAD}",
                "\N{EM QUAD}\N{EM QUAD}\N{EM QUAD}",
                "",
                "\N{EM QUAD}\N{EM QUAD}\N{EM QUAD}",
            )
        elif 0 in patterns_seen:  # Roller Coast
            em_quads = (
                "",
                "\N{EM QUAD}",
                "",
                "\N{EM QUAD}",
            )
        else:  # Huge and Small spike
            em_quads = (
                "",
                "",
                "",
                "",
            )

        spaces = f"\N{EM QUAD}{em_quads[pattern_num]}{pattern_space_table[pattern_num]}"
        return spaces


    @eCommands.command(name="graph_user",  # , aliases=["predict"],
                       brief="Graphs the possible outcomes for the week",
                       description="Graphs the possible outcomes for the week. You can also graph another users "
                                   "possible outcomes by including thier discord mention or User ID with the command.",
                       examples=["", "@Hibiki", "389590123654012632"],
                       hidden=True)
    async def graph_predict_prices_cmd_for_user_id(self, ctx: commands.Context, user_id: int):

        user = self.bot.get_user(user_id)
        display_name = user.display_name if user is not None else "Unknown"

        prices = await get_prices_for_user(self.bot.db_pool, user_id)
        if len(prices) == 0:
            embed = discord.Embed(title=f"Price predictions for {display_name}",
                                  description="\N{WARNING SIGN} Can not make a prediction as no prices have been recorded yet!")
            await ctx.send(embed=embed)
            return
        else:
            previous_pattern = await get_last_weeks_pattern_for_user(self.bot.db_pool, user_id)
            previous_pattern = previous_pattern or -1

            start = time.perf_counter()
            predictions, other_data = sm.get_predictions(prices, previous_pattern)
            log.info(f"Took: {time.perf_counter() - start} s")

            user_prediction = UserPredictions(user_id, display_name, predictions, other_data, previous_pattern)

            await self.graph_predict_prices(ctx, user, user_prediction)


    @eCommands.command(name="graph",#, aliases=["predict"],
                       brief="Graphs the possible outcomes for the week",
                       description="Graphs the possible outcomes for the week. You can also graph another users "
                                   "possible outcomes by including thier discord mention or User ID with the command.",
                       examples=["", "@Hibiki", "389590123654012632"])
    async def graph_predict_prices_cmd(self, ctx: commands.Context, user: Optional[discord.Member] = None):

        if user is None:
            user = ctx.author

        prices = await get_prices_for_user(self.bot.db_pool, user.id)
        if len(prices) == 0:
            embed = discord.Embed(title=f"Price predictions for {user.display_name}",
                                  description="\N{WARNING SIGN} Can not make a prediction as no prices have been recorded yet!")
            await ctx.send(embed=embed)
            return
        else:
            previous_pattern = await get_last_weeks_pattern_for_user(self.bot.db_pool, user.id)
            if previous_pattern is None:
                previous_pattern = -1

            start = time.perf_counter()
            predictions, other_data = sm.get_predictions(prices, previous_pattern)
            log.info(f"Took: {time.perf_counter() - start} s")

            user_prediction = UserPredictions(user.id, user.display_name, predictions, other_data, previous_pattern)

            await self.graph_predict_prices(ctx, user, user_prediction)


    async def graph_predict_prices(self, ctx: commands.Context, user: discord.Member, user_prediction: UserPredictions) -> discord.Message:

        embed = discord.Embed(title=f"Price predictions for {user_prediction.user_name}")

        predictions = user_prediction.patterns

        desc = f"With a previous week pattern of *{sm.pattern_definitions.name(user_prediction.last_pattern)}*," \
               f"\nYou have the following possible outcomes:\n"

        if len(predictions) == 0:
            desc += "**None!!!**\n**It is likely that your recorded price(s) are incorrect.**"
            image = None
        else:

            outcomes = defaultdict(list)
            guarantied_mins = defaultdict(list)
            outcome_txt = []
            patterns_seen = set()
            for pred in predictions:
                outcomes[pred.number].append(pred.weekMax)
                guarantied_mins[pred.number].append(pred.guaranteedMin)

                patterns_seen.add(pred.number)

            max_length = 0
            for pattern_num, prices in outcomes.items():
                pattern_name = sm.pattern_definitions.name(pattern_num)
                pattern_spaces = self.get_spaces_for_pattern(pattern_num, patterns_seen)

                if len(prices) == 1:
                    price_txt = f"Max Price Range: {min(guarantied_mins[pattern_num])} - {max(prices)}"
                    # pattern_desc = f"*{pattern_name}*{pattern_spaces} \n{(user_prediction.pattern_probability(pattern_num) * 100):.2f}% Chance w/ 1 Prediction"
                    chance_desc = f"{(user_prediction.pattern_probability(pattern_num) * 100):.2f}% Chance w/ 1 Prediction"

                else:
                    # price_txt = f"Max Prices: {min(prices)} - {max(prices)}"
                    price_txt = f"Max Price Range: {min(guarantied_mins[pattern_num])} - {max(prices)}"
                    # pattern_desc = f"*{pattern_name}*{pattern_spaces} \n{(user_prediction.pattern_probability(pattern_num) * 100):.2f}% Chance w/ {len(prices)} Predictions"
                    chance_desc = f"{(user_prediction.pattern_probability(pattern_num) * 100):.2f}% Chance w/ {len(prices)} Predictions"

                max_length = len(price_txt) if len(price_txt) > max_length else max_length
                outcome_txt.append((price_txt, chance_desc, pattern_name))#pattern_desc))

                # embed.add_field(name="\N{Zero Width Space}",
                #                 value=f"__*{pattern_name}*__\n{price_txt}\n{chance_desc}",
                #                 inline=False)

            for price_txt, chance_desc, pattern_name in outcome_txt:
                embed_desc = '{0}`{1:<{width}}`\N{EM QUAD}{2}\n'.format(0 * ' ', price_txt, chance_desc, width=max_length)

                # embed.add_field(name="\N{Zero Width Space}",
                #                 value=f"__*{pattern_name}*__\n{embed_desc}",  #f"__*{pattern_name}*__\n{price_txt}\n{chance_desc}",
                #                 inline=False)
                desc += f"__*{pattern_name}*__\n{embed_desc}"


            # for price_txt, pattern_desc in outcome_txt:
            #
            #     desc += '{0}`{1:<{width}}`\N{EM QUAD}{2}\n'.format(0 * ' ', price_txt, pattern_desc, width=max_length)

            image_buffer = matplotgraph_predictions(ctx.author, predictions, user_prediction.min_max, user_prediction.expected_prices)#average_prices)
            image_buffer.seek(0)
            image = discord.File(filename="turnipChart.png", fp=image_buffer)
            embed.set_image(url=f"attachment://turnipChart.png")
            log.info("Generated Graph")

        # Make sure we don't exceed the max char limit of the description field.
        if len(desc) > 2000:
            desc = desc[:1996] + "..."

        embed.description = desc

        msg = await ctx.send(embed=embed, file=image)
        return msg


    @commands.guild_only()
    @eCommands.command(name="predict",  # aliases=["list_prices"],
                       brief="Predict who on the server will have the highest prices for the week.",
                       examples=[""])
    async def guild_predict(self, ctx: commands.Context):
        guild: discord.Guild = ctx.guild
        embed = discord.Embed(title="Users With The Highest Potential Prices")

        user_predictions = await get_guild_user_predictions(self.bot.db_pool, guild=guild)

        if len(user_predictions) > 0:

            for i, pred in enumerate(user_predictions[:sm.max_guild_predictions]):
                pattern_count = pred.prediction_count()
                embed.add_field(name=" ‌‌‌",
                                value=f"{number_emotes[i+1]} <@{pred.user_id}>\n"
                                      f"Most Likely Max Price: **{max(pred.expected_prices):.2f}**\n"
                                      f"Max Price Range: **{pred.min_max.guaranteedMin} - {pred.min_max.weekMax}**\n"
                                      f"Most Likely Pattern: **{pred.best().description}** {pred.pattern_probability(pred.best().number)*100:.2f}% Chance",#( {pattern_count[pred.best().number]}/{sum(pattern_count)} predictions)",
                                inline=False)

            image_buffer = matplotgraph_guild_predictions(user_predictions)
            image = None
            if image_buffer is not None:
                image_buffer.seek(0)
                image = discord.File(filename="turnipGuildChart.png", fp=image_buffer)
                embed.set_image(url=f"attachment://turnipGuildChart.png")
        else:
            # If we can't make a prediction, say so and bail
            embed.description = "\N{WARNING SIGN} No predictions can currently be made!"
            await ctx.send(embed=embed)
            return

        buttons = [
            (number_emotes[1], "one"),
            (number_emotes[2], "two"),
            (number_emotes[3], "three"),
        ]

        buttons = buttons[:len(user_predictions)]  # Make sure we don't have more buttons than predictions
        predict_ui = StringReactPage(embed=embed, image=image, edit_in_place=False, remove_msgs=False, buttons=buttons,
                                     allowable_responses=[], cancel_btn=False)

        specific_prediction_msg = None
        first_run = True
        while True:
            response = await predict_ui.run(ctx, send_new_msg=first_run)
            first_run = False

            if response is None:

                await predict_ui.finish()
                return

            elif response.content() == "one":
                user_pred = user_predictions[0]
                d_member: discord.Member = guild.get_member(user_pred.user_id)
                if specific_prediction_msg is not None:
                    await specific_prediction_msg.delete(delay=1)
                specific_prediction_msg = await self.graph_predict_prices(ctx, d_member, user_pred)

            elif response.content() == "two":
                user_pred = user_predictions[1]
                d_member: discord.Member = guild.get_member(user_pred.user_id)
                if specific_prediction_msg is not None:
                    await specific_prediction_msg.delete(delay=1)
                specific_prediction_msg = await self.graph_predict_prices(ctx, d_member, user_pred)

            elif response.content() == "three":
                user_pred = user_predictions[2]
                d_member: discord.Member = guild.get_member(user_pred.user_id)
                if specific_prediction_msg is not None:
                    await specific_prediction_msg.delete(delay=1)
                specific_prediction_msg = await self.graph_predict_prices(ctx, d_member, user_pred)


    @commands.is_owner()
    @eCommands.command(name="tg",  # aliases=["list_prices"],
                       brief="test graPH",
                       examples=[""], hidden=True)
    async def TESTGRAPH(self, ctx: commands.Context):
        import plotly.io as pio
        pio.orca.config.use_xvfb = True
        import plotly.graph_objects as go
        import numpy as np
        np.random.seed(1)

        N = 100
        x = np.random.rand(N)
        y = np.random.rand(N)
        colors = np.random.rand(N)
        sz = np.random.rand(N) * 30

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=go.scatter.Marker(
                size=sz,
                color=colors,
                opacity=0.6,
                colorscale="Viridis"
            )
        ))

        # fig.show()
        # fig.show(renderer="png")
        fig.write_image("fig1.png")

        # await ctx.send("Done!")





def setup(bot):
    bot.add_cog(StalkMarket(bot))
