"""

"""
import sys
import logging
import traceback
# from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Union

import discord
from discord.ext import commands, tasks
import asyncpg

import db
# from utils.errors import handle_permissions_error
# from miscUtils import log_error_msg

log = logging.getLogger(__name__)

extensions = (
    # -- Command Extensions -- #
    'cogs.stalkMarket',
    'cogs.helpCmd',
    'cogs.utilities',
)


class GGBot(commands.Bot):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_pool: Optional[asyncpg.pool.Pool] = None
        self.config: Optional[Dict] = None

        self.update_playing.start()


    def load_cogs(self):
        for extension in extensions:
            try:
                self.load_extension(extension)
                log.info(f"Loaded {extension}")
            except Exception as e:
                log.info(f'Failed to load extension {extension}.', file=sys.stderr)
                traceback.print_exc()


    # async def send_log(self, log_ch: discord.TextChannel, event_type: str, embed: Optional[discord.Embed] = None, file: Optional[discord.File] = None) -> discord.Message:
    #     log.info(f"sending {event_type} to {log_ch.name}")
    #     try:
    #         msg = await log_ch.send(embed=embed, file=file)
    #         return msg
    #     except discord.Forbidden as e:
    #         await handle_permissions_error(self, log_ch, event_type, e, None)


    # region Now Playing Update Task Methods
    # noinspection PyCallingNonCallable
    @tasks.loop(minutes=30)
    async def update_playing(self):
        log.info("Updating now Playing...")
        await self.set_playing_status()


    @update_playing.before_loop
    async def before_update_playing(self):
        await self.wait_until_ready()


    async def set_playing_status(self):
        activity = discord.Game("{}help | in {} Servers".format(self.command_prefix, len(self.guilds)))
        await self.change_presence(status=discord.Status.online, activity=activity)

    # endregion


    async def get_channel_safe(self, channel_id: int) -> Optional[discord.TextChannel]:
        channel = self.get_channel(channel_id)
        if channel is None:
            log.info("bot.get_channel failed. Querying API...")
            try:
                channel = await self.fetch_channel(channel_id)
            except discord.NotFound:
                return None
        return channel



