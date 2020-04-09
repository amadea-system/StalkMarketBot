"""
Cog containing various utility commands.
Commands include:
    bot_invite
    ping
    stats
    verify_perm
    purge

Part of Stalk Market Bot.
"""


import os
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple, NamedTuple

import psutil
import discord
from discord.ext import commands

from utils.paginator import FieldPages
import db

if TYPE_CHECKING:
    from bot import VBot

log = logging.getLogger(__name__)


class Utilities(commands.Cog):
    def __init__(self, bot: 'VBot'):
        self.bot = bot

    # region Invite Command
    @commands.group(name='invite',
                    aliases=['bot_invite'],
                    brief='Get an invite for Stalk Market Bot.',
                    description='Get an invite for Stalk Market Bot.')
    async def invite_link(self, ctx: commands.Context):

        if ctx.invoked_subcommand is None:
            perm = discord.Permissions(
                add_reactions=True,     # Required for the reaction based menu system.
                read_messages=True,     # Required to delete messages from void channels & and to see commands being used.
                manage_messages=True,   # Optional for the reaction based menu system. (Needed so we can remove the reactions from menus after a user clicks on a react & when the menu times out.) (New)
                embed_links=True,       # Required for configuration of the bot.
                read_message_history=True,  # Required for the reaction based menu system. (Can not react to a message that has reactions by other users with out this permission.)
                send_messages=True,     # Required for the log messages and the configuration of the bot.
            )

            link = discord.utils.oauth_url(self.bot.user.id, permissions=perm)
            await ctx.send(f"Here's a link to invite Stalk Market Bot to your server:\n{link}\n\n")

    # @invite_link.command(name='explain', brief="Explains why Gabby Gums needs the permissions requested.")
    # async def invite_link_explain(self, ctx: commands.Context):
    #     embed = discord.Embed(title="Breakdown Of Requested Permissions",
    #                           description="Here is a breakdown of all the permissions Gabby Gums asks for and why.\n"
    #                                       "If you have any additional questions regarding these permissions, feel free to stop by our support server: https://discord.gg/3Ugade9\n",
    #                           color=discord.Color.from_rgb(80, 135, 135))
    #
    #
    #     embed.add_field(name="Read Messages (Required)",
    #                     value="The Read Messages permission is required so that Gabby Gums can do pretty much anything at all.",
    #                     inline=False)
    #
    #     embed.add_field(name="Embed Links (Required)",
    #                     value="The Embed Links permission is required to send Embeds (Like the one this explination is in).\n"
    #                           "Without it, no logs can be sent and most commands will not be operational.",
    #                     inline=False)
    #
    #     embed.add_field(name="Send Messages (Required)",
    #                     value="The Send Messages permission is required to send any logs and to respond to any commands.\n",
    #                     inline=False)
    #
    #     embed.add_field(name="Use External Emojis (Required)",
    #                     value="The Use External Emojis permission is required as some logs and configuration menus use external emojis in them.",
    #                     inline=False)
    #
    #     embed.add_field(name="Add Reactions (Required)",
    #                     value="The Add Reactions permission is required for the reaction based menu system that some commands of use.",
    #                     inline=False)
    #
    #     embed.add_field(name="Read Message History (Mostly Required)",
    #                     value="The Read Message History permission is required for the reaction based menu system in some cases.\n"
    #                           "Additionally, it is also needed for the *Archive* command to function.",
    #                     inline=False)
    #     await ctx.send(embed=embed)
    # endregion


    @commands.guild_only()
    @commands.command(name='ping', aliases=['pong'],
                      brief='Shows the current bot latency.',
                      description='Shows the current bot latency.')
    async def ping_command(self, ctx: commands.Context):

        db_start = time.perf_counter()
        await db.get_all_accounts_for_guild(self.bot.db_pool, ctx.guild.id)  # test DB speed
        db_end = time.perf_counter()

        embed = discord.Embed(title="Pinging...", description=" \n ", color=0x000000)
        start = time.perf_counter()
        # Gets the timestamp when the command was used

        msg = await ctx.send(embed=embed)
        # Sends a message to the user in the channel the message with the command was received.
        # Notifies the user that pinging has started
        new_embed = discord.Embed(title="Pong!",
                                  description="Round trip messaging time: **{:.2f} ms**. \nAPI latency: **{:.2f} ms**.\nDatabase latency: **{:.2f} ms**".
                                  format((time.perf_counter() - start) * 1000, self.bot.latency * 1000,
                                         (db_end - db_start) * 1000), color=0x000000)
        await msg.edit(embed=new_embed)


    @commands.command(name='stats', aliases=['stat', 'top'],
                      brief='Shows stats such as CPU, memory usage, disk space usage.',
                      description='Shows various stats such as CPU, memory usage, disk space usage, and more.')
    async def stats_command(self, ctx: commands.Context):

        def folder_size(path='.'):
            total = 0
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += folder_size(entry.path)
            return total

        pid = os.getpid()
        py = psutil.Process(pid)
        memory_use = py.memory_info()[0] / 1024 / 1024
        disk_usage = psutil.disk_usage("/")
        disk_space_free = disk_usage.free / 1024 / 1024
        disk_space_used = disk_usage.used / 1024 / 1024
        disk_space_percent_used = disk_usage.percent

        try:
            # noinspection PyUnresolvedReferences
            load_average = os.getloadavg()
        except AttributeError:  # Get load avg is not available on windows
            load_average = [-1, -1, -1]

        embed = discord.Embed(title="CPU and memory usage:",
                              description="CPU: **{}%** \nLoad average: **{:.2f}, {:.2f}, {:.2f}**\nMemory: **{:.2f} MB**"
                                          "\nDisk space: **{:.2f} MB Free**, **{:.2f} MB Used**, **{}% Used**\n# of guilds: **{}**".
                              format(psutil.cpu_percent(), load_average[0], load_average[1], load_average[2],
                                     memory_use, disk_space_free, disk_space_used, disk_space_percent_used, len(self.bot.guilds)), color=0x000000)

        await ctx.send(embed=embed)

    # region Permissions Verification Command
    # @commands.command(name="permissions",
    #                   aliases=["verify_permissions", "perm", "permissions_check", "perm_check", "verify_perm"],
    #                   brief="Checks for any permissions or configuration problems.",
    #                   description="Checks for any possible permission or configuration problems that could interfere with the operations of Stalk Market Bot",
    #                   )
    # async def verify_permissions(self, ctx: commands.Context, guild_id: Optional[str] = None):
    #
    #     # Current number of fields:  Perm: 8/25, Conf: 5/25
    #
    #     if guild_id is not None:
    #         guild: discord.Guild = self.bot.get_guild(int(guild_id.strip()))
    #     else:
    #         guild: discord.Guild = ctx.guild
    #
    #     if guild is None:
    #         if guild_id is None:
    #             await ctx.send("A Guild ID is required in DMs")
    #         else:
    #             await ctx.send("{} is an invalid guild ID".format(guild_id))
    #         return
    #
    #     # ToDO: check for send permissions for ctx and log error if unavailable.
    #     perm_embed = discord.Embed(title="Permissions Debug for {}".format(guild.name), color=0x000000)
    #
    #     perms = {'read': [], 'send': [], 'manage_messages': [], 'embed_links': [], "read_msg_history": []}
    #
    #     errors_found = False
    #
    #     for channel in guild.channels:
    #         channel: discord.TextChannel
    #         permissions: discord.Permissions = channel.guild.me.permissions_in(channel)
    #
    #         if channel.type == discord.ChannelType.text:
    #
    #
    #             if discord.utils.get(void_channels, channel_id=channel.id) is not None:
    #
    #                 if permissions.read_messages is False:
    #                     errors_found = True
    #                     perms['read'].append(f"<#{channel.id}>")
    #
    #                 # if permissions.send_messages is False:
    #                 #     errors_found = True
    #                 #     perms['send'].append(f"<#{channel.id}>")
    #                 #
    #                 # if permissions.embed_links is False:  # Needed to actually send the embed messages
    #                 #     errors_found = True
    #                 #     perms['embed_links'].append(f"<#{channel.id}>")
    #                 #
    #                 if permissions.manage_messages is False:  # Needed for Channel Update logs and some commands
    #                     errors_found = True
    #                     perms["manage_messages"].append(f"<#{channel.id}>")
    #                 #
    #                 # if permissions.read_msg_history is False:  # Needed for Bulk Delete logs and the archive command
    #                 #     errors_found = True
    #                 #     perms["read_msg_history"].append(f"<#{channel.id}>")
    #
    #     if len(perms['read']) > 0:
    #         send_msg = "\N{Warning Sign} CRITICAL ERROR!! The following channels are configured as void channels but are missing the **Read Messages** permission.\n" \
    #                    "As such, no messages can be deleted from:\n"
    #         send_msg = send_msg + "\n".join(perms['read'])
    #         perm_embed.add_field(name="Missing Read Messages Permissions in Void Channels", value=f"{send_msg}\n\N{ZERO WIDTH NON-JOINER}",
    #                              inline=False)
    #
    #     if len(perms['manage_messages']) > 0:
    #         send_msg = "\N{Warning Sign} CRITICAL ERROR!! The following channels are configured as void channels but are missing the **Manage Messages** permission.\n" \
    #                    "As such no messages can be deleted from:\n"
    #         send_msg = send_msg + "\n".join(perms['manage_messages'])
    #         perm_embed.add_field(name="Missing Manage Messages Permission in Void Channels", value=f"{send_msg}\n\N{ZERO WIDTH NON-JOINER}",
    #                              inline=False)
    #
    #     # Set the appropriate embed description
    #     if errors_found:
    #         perm_embed.description = "Uh oh! Problems were found!"
    #     else:
    #         perm_embed.description = "No problems found!"
    #
    #     await ctx.send(embed=perm_embed)
    # endregion

    # region DB Performance Statistics Command
    @commands.cooldown(rate=1, per=10, type=commands.BucketType.default)
    @commands.command(aliases=["db_stats", "db_performance"],
                      brief="Shows various time stats for the database.",)
    async def db_perf(self, ctx: commands.Context):

        embed_entries = []
        stats = db.db_perf.stats()

        for key, value in stats.items():
            # Don't bother showing stats for one offs
            if key != 'create_tables' and key != 'migrate_to_latest':
                header = f"{key}"

                msg_list = []
                for sub_key, sub_value in value.items():
                    if sub_key == "calls":
                        msg_list.append(f"{sub_key}: {sub_value:.0f}")
                    else:
                        msg_list.append(f"{sub_key}: {sub_value:.2f}")

                if len(msg_list) > 0:
                    msg = "\n".join(msg_list)
                    embed_entries.append((header, msg))

        page = FieldPages(ctx, entries=embed_entries, per_page=15)
        page.embed.title = f"DB Statistics:"
        await page.paginate()
    # endregion

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):

        if 'error_log_channel' not in self.bot.config:
            return
        error_log_channel = self.bot.get_channel(self.bot.config['error_log_channel'])

        if isinstance(error, commands.CommandOnCooldown):
            # DDOS Protection. Send alerts in the error log if we are potentially being DDOSed with resource intensive commands.
            # Only in this cog atm as these are the high risk items.
            await error_log_channel.send(f"⚠ Excessive use of {ctx.command.module}.{ctx.command.name} by <@{ctx.author}> ({ctx.author.id}) in {ctx.guild} ⚠ ")
            return


def setup(bot):
    bot.add_cog(Utilities(bot))
