"""
Cog implementing discord interface of Stalk Market Predictions
Part of Stalk Market Bot.
"""

import logging
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple, NamedTuple, Type, Any
from datetime import datetime

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

from utils.uiElements import BoolPage, StringReactPage


if TYPE_CHECKING:
    from bot import GGBot

log = logging.getLogger(__name__)

day_segment_names = ["Sunday Buy Price", "N/A", "Monday Morning", "Monday Afternoon", "Tuesday Morning",
                     "Tuesday Afternoon", "Wednesday Morning", "Wednesday Afternoon", "Thursday Morning",
                     "Thursday Afternoon", "Friday Morning", "Friday Afternoon", "Saturday Morning", "Saturday Afternoon"]

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

        est = timezone('US/Eastern')
        # now = datetime.utcnow()
        now = est.fromutc(datetime.utcnow())

        day_of_week = int(now.strftime("%w"))
        if now.hour >= 12:  # Past noon.
            day_segment = day_of_week * 2 + 1
        else:
            day_segment = day_of_week * 2

        if day_segment == 1:
            day_segment = 0
            # await ctx.send(f"Error! You can not set a price for Sunday Afternoon!")
            # return

        week_of_year = int(now.strftime("%U"))  # TODO: account for begining of the year
        year = now.year
        await self.add_new_price_handler(ctx, price, year, week_of_year, day_segment)


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

        now = datetime.utcnow()
        day_segment = 2

        week_of_year = int(now.strftime("%U"))  # TODO: account for begining of the year
        year = now.year

        embed = discord.Embed(title=f"Add New Prices", description="Are you sure you want to add the following prices?")

        for price in parsed_prices:
            if price is not None:
                embed.add_field(name=day_segment_names[day_segment], value=f"{price} Bells")
            day_segment += 1

        confirmation = BoolPage(embed=embed)

        response = await confirmation.run(ctx)
        if response:
            day_segment = 2
            for price in parsed_prices:
                if price is not None:
                    new_price = db.Prices(user_id=ctx.author.id, account_id=0, year=year, week=week_of_year,
                                          day_segment=day_segment, price=price)
                    await db.add_price(self.bot.db_pool, new_price)
                day_segment += 1

            await ctx.send(f"Prices set!")
        else:
            await ctx.send(f"Canceled!")


    @eCommands.command(name="add_price_at", aliases=["add_at", "ap_at"],
                       brief="Add a new price at a specific date & time.",
                       examples=['39 1 day and 6 hours ago utc+1', "42 4/5 11:00am est"])
    async def add_new_price_at(self, ctx: commands.Context, price: int, *, date: str):
        await db.add_account(self.bot.db_pool, ctx.guild.id, ctx.author.id, 0, "")

        now = dateparser.parse(date)
        if now is None:
            await ctx.send(f"Error! Unable to determine when {date} is!")
            return

        day_of_week = int(now.strftime("%w"))
        if now.hour >= 12:  # Past noon.
            day_segment = day_of_week * 2 + 1
        else:
            day_segment = day_of_week * 2

        if day_segment == 1:
            day_segment = 0

        week_of_year = int(now.strftime("%U"))  # TODO: account for begining of the year
        year = now.year

        await self.add_new_price_handler(ctx, price, year, week_of_year, day_segment)


    @eCommands.command(name="list", aliases=["list_prices"],
                       brief="Lists the current weeks recorded prices.",
                       examples=[""])
    async def list_prices(self, ctx: commands.Context):

        prices = await self.get_prices(ctx.author.id)

        embed = discord.Embed(title=f"Prices for {ctx.author.display_name}")
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
        est = timezone('US/Eastern')
        # now = datetime.utcnow()
        now = est.fromutc(datetime.utcnow())

        day_of_week = int(now.strftime("%w"))
        if now.hour >= 12:  # Past noon.
            day_segment = day_of_week * 2 + 1
        else:
            day_segment = day_of_week * 2

        if day_segment == 1:
            day_segment = 0
            # await ctx.send(f"Error! You can not set a price for Sunday Afternoon!")
            # return

        week_of_year = int(now.strftime("%U"))  # TODO: account for begining of the year
        year = now.year

        embed = discord.Embed(title="Remove Price",
                              description=f"Do you wish to remove the recorded price for {day_segment_names[day_segment]}?")

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
                                           description=f"The price of for {day_segment_names[day_segment]} was removed.")
                await remove_prompt.finish(last_embed)

                remove_price = db.Prices(user_id=ctx.author.id, account_id=0, year=year, week=week_of_year,
                                         day_segment=day_segment, price=0)
                await db.remove_price(self.bot.db_pool, remove_price)

                return

            elif response.content() == "left":
                if day_segment == 0:
                    pass
                elif day_segment == 2:
                    day_segment = 0  # Skip 1
                else:
                    day_segment -= 1  # Decrement by 1

                remove_prompt.embed = discord.Embed(title="Remove Price",
                                                    description=f"Do you wish to remove the recorded price for {day_segment_names[day_segment]}?")

            elif response.content() == "right":

                if day_segment == 0:
                    day_segment = 2  # Skip 1
                elif day_segment == 13:
                    pass
                else:
                    day_segment += 1  # Increment by 1

                remove_prompt.embed = discord.Embed(title="Remove Price",
                                                    description=f"Do you wish to remove the recorded price for {day_segment_names[day_segment]}?")


    @commands.is_owner()
    @eCommands.command(name="tpredict", #aliases=["list_prices"],
                       brief="Predicts the possible outcomes",
                       examples=[""], hidden=True)
    async def predict_prices(self, ctx: commands.Context):

        embed = discord.Embed(title=f"Price predictions for {ctx.author.display_name}")

        prices = await self.get_prices(ctx.author.id)
        # predictions, min_max = sm.get_test_predictions()
        predictions, min_max, average_prices = sm.get_predictions(prices)

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


    @eCommands.command(name="graph",#, aliases=["predict"],
                       brief="Graphs the possible outcomes for the week",
                       description="Graphs the possible outcomes for the week. You can also graph another users "
                                   "possible outcomes by including thier discord mention or User ID with the command.",
                       examples=["", "@Hibiki", "389590123654012632"])
    async def graph_predict_prices(self, ctx: commands.Context, user: Optional[discord.Member] = None):

        if user is None:
            user = ctx.author

        embed = discord.Embed(title=f"Price predictions for {user.display_name}")

        prices = await self.get_prices(user.id)
        # predictions, min_max = sm.get_test_predictions()
        predictions, min_max, average_prices = sm.get_predictions(prices)

        desc = f"You have the following possible outcomes:"

        if len(predictions) == 0:
            desc += "\n**None!!!**\n**It is likely that the dataset is incorrect.**"
            image = None
        else:
            outcomes = []
            for pred in predictions:
                outcome = f"\n**Max Price: {pred.weekMax}** {pred.description}"
                if outcome not in outcomes:
                    desc += outcome
                outcomes.append(outcome)
                # mentioned_pred.append(pred.description)

            image_buffer = sm.matplotgraph_predictions(ctx.author, predictions, min_max, average_prices)
            image_buffer.seek(0)
            image = discord.File(filename="turnipChart.png", fp=image_buffer)
            embed.set_image(url=f"attachment://turnipChart.png")
            log.info("Generated Graph")

        if len(desc) > 2000:
            desc = desc[:1996] + "..."

        embed.description = desc

        await ctx.send(embed=embed, file=image)


    async def get_prices(self, user_id: int, date=None) -> List[db.Prices]:
        if date is None:
            # date = datetime.utcnow()
            est = timezone('US/Eastern')
            date = est.fromutc(datetime.utcnow())

        week_of_year = int(date.strftime("%U"))  # TODO: account for begining of the year
        year = date.year

        prices = await db.get_prices(self.bot.db_pool, user_id, 0, year, week_of_year)
        prices.sort(key=lambda x: x.day_segment)

        return prices


    @commands.guild_only()
    @eCommands.command(name="predict",  # aliases=["list_prices"],
                       brief="Predict who on the server will have the highest prices for the week.",
                       examples=[""])
    async def guild_predict(self, ctx: commands.Context):

        users = await db.get_all_accounts_for_guild(self.bot.db_pool, ctx.guild.id)

        user_predictions = []
        for user in users:
            prices = await self.get_prices(user.user_id)

            predictions, min_max, average_prices = sm.get_predictions(prices)
            user_predictions.append((user.user_id, predictions, min_max, average_prices))


        def sort_func(possibility):
            return max(possibility[3])

        user_predictions.sort(reverse=True, key=sort_func)

        embed = discord.Embed(title="Users With The Highest Potential Prices")

        for pred in user_predictions:
            embed.add_field(name=" ‌‌‌",
                            value=f"<@{pred[0]}>\n"
                                  f"Max Average Price: {max(pred[3])}\n"
                                  f"Max Possible Price: {pred[2].weekMax}\n",
                            inline=False)

        await ctx.send(embed=embed)


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
