'''

'''

import json
import logging
import traceback
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Union

import asyncpg
import discord
from discord.ext import commands

import db



from bot import GGBot


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

bot = GGBot(command_prefix="t;",
               max_messages=100000,
               # description="A simple logging bot that ignores PluralKit proxies.\n",
               owner_id=389590659335716867,
               case_insensitive=True)


@bot.event
async def on_ready():
    logging.info('Connected using discord.py version {}!'.format(discord.__version__))
    logging.info('Username: {0.name}, ID: {0.id}'.format(bot.user))
    logging.info("Connected to {} servers.".format(len(bot.guilds)))
    logging.info('------')

    logging.warning("Stalk Market is fully loaded.")


# ---- Command Error Handling ----- #
@bot.event
async def on_command_error(ctx, error):

    # https://gist.github.com/EvieePy/7822af90858ef65012ea500bcecf1612
    # This prevents any commands with local handlers being handled here in on_command_error.
    if hasattr(ctx.command, 'on_error'):
        return

    if type(error) == discord.ext.commands.NoPrivateMessage:
        await ctx.send("⚠ This command can not be used in DMs!!!")
        return
    elif type(error) == discord.ext.commands.CommandNotFound:
        await ctx.send("⚠ Invalid Command!!!")
        return
    elif type(error) == discord.ext.commands.MissingPermissions:
        await ctx.send("⚠ You need the **Manage Messages** permission to use this command".format(error.missing_perms))
        return
    elif type(error) == discord.ext.commands.MissingRequiredArgument:
        await ctx.send("⚠ {}".format(error))
    elif type(error) == discord.ext.commands.BadArgument:
        await ctx.send("⚠ {}".format(error))
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send("⚠ {}".format(error))
    else:
        # await ctx.send("⚠ {}".format(error))
        raise error


@bot.event
async def on_error(event_name, *args):
    logging.exception("Exception from event {}".format(event_name))

    if 'error_log_channel' not in bot.config:
        return

    error_log_channel = bot.get_channel(bot.config['error_log_channel'])

    embed = None
    # Determine if we can get more info, otherwise post without embed
    if args and type(args[0]) == discord.Message:
        message: discord.Message = args[0]
        # embeds.exception_w_message(message)
    elif args and type(args[0]) == discord.RawMessageUpdateEvent:
        logging.error("After Content:{}.".format(args[0].data['content']))
        if args[0].cached_message is not None:
            logging.error("Before Content:{}.".format(args[0].cached_message.content))
    # Todo: Add more

    traceback_message = "```python\n{}```".format(traceback.format_exc())
    traceback_message = (traceback_message[:1993] + ' ...```') if len(traceback_message) > 2000 else traceback_message
    await error_log_channel.send(content=traceback_message, embed=embed)


if __name__ == '__main__':

    with open('config.json') as json_data_file:
        config = json.load(json_data_file)

    db_pool: asyncpg.pool.Pool = asyncio.get_event_loop().run_until_complete(db.create_db_pool(config['db_uri']))
    asyncio.get_event_loop().run_until_complete(db.create_tables(db_pool))

    bot.config = config
    bot.db_pool = db_pool
    bot.command_prefix = config['bot_prefix']

    bot.load_cogs()
    bot.run(config['token'])

    logging.info("cleaning Up and shutting down")
