"""
General helper functions for void.
Function abilities include:
    Functions for handling long text

Part of Stalk Market Bot.
"""

import sys
import time
import statistics as stats
import string
import logging
import traceback
import functools

from collections import defaultdict


from typing import Union, Optional, List, TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from bot import VBot

log = logging.getLogger(__name__)


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

            loop_stats['sum'] = sum(value)

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


perf_stats = DBPerformance()


def async_perf_timer(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        response = await func(*args, **kwargs)
        end_time = time.perf_counter()

        # if len(args) > 1:
        #     logging.info("DB Query {} from {} in {:.3f} ms.".format(func.__name__, args[1], (end_time - start_time) * 1000))
        # else:
        # logging.info("Func {} ran in {:.3f} ms.".format(func.__name__, (end_time - start_time) * 1000))
        perf_stats.time[func.__name__].append((end_time - start_time) * 1000)
        return response

    return wrapper


def perf_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        response = func(*args, **kwargs)
        end_time = time.perf_counter()

        # if len(args) > 1:
        #     logging.info("DB Query {} from {} in {:.3f} ms.".format(func.__name__, args[1], (end_time - start_time) * 1000))
        # else:
        # logging.info("Func {} ran in {:.3f} ms.".format(func.__name__, (end_time - start_time) * 1000))
        perf_stats.time[func.__name__].append((end_time - start_time) * 1000)
        return response

    return wrapper


async def send_long_msg(channel: [discord.TextChannel, commands.Context], message: str, code_block: bool = False, code_block_lang: str = "python"):

    if code_block:
        if len(code_block_lang) > 0:
            code_block_lang = code_block_lang + "\n"
        code_block_start = f"```{code_block_lang}"
        code_block_end = "```"
        code_block_extra_length = len(code_block_start) + len(code_block_end)
        chunks = split_text(message, max_size=2000 - code_block_extra_length)
        message_chunks = [code_block_start + chunk + code_block_end for chunk in chunks]

    else:
        message_chunks = split_text(message, max_size=2000)

    for chunk in message_chunks:
        await channel.send(chunk)


def split_text(text: Union[str, List], max_size: int = 2000, delimiter: str = "\n") -> List[str]:
    """Splits the input text such that no entry is longer that the max size """
    delim_length = len(delimiter)

    if isinstance(text, str):
        if len(text) < max_size:
            return [text]
        text = text.split(delimiter)
    else:
        if sum(len(i) for i in text) < max_size:
            return ["\n".join(text)]

    output = []
    tmp_str = ""
    count = 0
    for fragment in text:
        fragment_length = len(fragment) + delim_length
        if fragment_length > max_size:
            raise ValueError("A single line exceeded the max length. Can not split!")  # TODO: Find a better way than throwing an error.
        if count + fragment_length > max_size:
            output.append(tmp_str)
            tmp_str = ""
            count = 0

        count += fragment_length
        tmp_str += f"{fragment}{delimiter}"

    output.append(tmp_str)

    return output


async def log_error_msg(bot: 'VBot', error_messages: Optional[Union[str, List[str], Exception]], header: Optional[str] = None, code_block: bool = False) -> bool:
    """
    Attempts to send a message to the Global Error Discord Channel.

    Returns False if the error_log_channel is not defined in the Config,
        if the error_log_channel can not be resolved to an actual channel, or if the message fails to send.

    Returns True if successful.
    """

    if 'error_log_channel' not in bot.config:
        return False  # No error log channel defined in config, can not log

    # Check to see if there was an error message passed and bail if there wasn't
    if error_messages is None:
        return True  # Should this be True? False isn't really accurate either....
    # If list is empty, return
    elif isinstance(error_messages, list):  # If type is list

        if len(error_messages) == 0:  # List is empty. Bail
            return True  # Should this be True? False isn't really accurate either....
        # Convert it into a single string.
        error_messages = "\n".join(error_messages)
    elif isinstance(error_messages, Exception):
        error_messages = full_stack()
        code_block = True  # Override code block for exceptions.
    else:
        if error_messages == "":  # Empty
            return True  # Should this be True? False isn't really accurate either....

    # Try to get the channel from discord.py.
    error_log_channel = bot.get_channel(bot.config['error_log_channel'])
    if error_log_channel is None:
        return False

    # If the header option is used, include the header message at the front of the message
    if header is not None:
        error_messages = f"{header}\n{error_messages}"
    # Attempt to send the message
    try:
        await send_long_msg(error_log_channel, error_messages, code_block=code_block)
        return True
    except discord.DiscordException as e:
        log.exception(f"Error sending log to Global Error Discord Channel!: {e}")
        return False


def full_stack():
    exc = sys.exc_info()[0]
    if exc is not None:
        f = sys.exc_info()[-1].tb_frame.f_back
        stack = traceback.extract_stack(f, limit=5)
    else:
        stack = traceback.extract_stack(limit=5)[:-1]  # last one would be full_stack()
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


def prettify_permission_name(perm_name: str) -> str:
    pretty_perm_name = string.capwords(f"{perm_name}".replace('_', ' '))  # Capitalize the permission names and replace underlines with spaces.
    pretty_perm_name = "Send TTS Messages" if pretty_perm_name == "Send Tts Messages" else pretty_perm_name  # Mak sure that we capitalize the TTS acronym properly.
    return pretty_perm_name

