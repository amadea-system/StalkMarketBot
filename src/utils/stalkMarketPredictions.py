"""
Logic for Stalk Market Predictions
Part of Stalk Market Bot.
"""

import math
import logging

from io import BytesIO
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple, NamedTuple, Any

import discord
from discord.utils import get

# import plotly.graph_objects as go
# from plotly.colors import DEFAULT_PLOTLY_COLORS

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
# from scipy.ndimage.filters import gaussian_filter1d
# from scipy.interpolate import splrep, splev, splprep
import numpy as np

from db import Prices

log = logging.getLogger(__name__)


day_segment_names = ["Sunday Buy Price", "N/A", "Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM", "Wed PM",
                     "Thu AM", "Thu PM", "Fri AM", "Fri PM", "Sat AM", "Sat PM"]


class InconsistentPhaseLengths(Exception):
    pass


def minimum_rate_from_given_and_base(given_price: int, buy_price: int) -> float:
    return 10000 * (given_price - 1) / buy_price


def maximum_rate_from_given_and_base(given_price: int, buy_price: int) -> float:
    return 10000 * given_price / buy_price


class MinMaxPrice:

    def __init__(self, _min: int, _max: int, actual: Optional[int] = None):
        self.min = _min
        self.max = _max
        self.actual = actual

    def __str__(self):
        if self.is_actual_price():
            return f"{self.actual}"
        else:
            return f"{self.min} - {self.max}"

    def is_actual_price(self) -> bool:
        # return self.max == self.min
        return self.actual is not None


class Pattern:

    def __init__(self, description: str, number: int, prices: List[MinMaxPrice]):
        self.prices = prices
        self.number = number
        self.description = description

        self.weekMin = 999
        self.weekMax = 0
        self.calculate_min_max()


    def calculate_min_max(self):

        week_mins = []
        week_maxes = []

        for day in self.prices[2:]:
            week_mins.append(day.min)
            week_maxes.append(day.max)

        self.weekMin = min(week_mins)
        self.weekMax = max(week_maxes)


def generate_pattern_0_with_lengths(given_prices: List[int], high_phase_1_len, dec_phase_1_len, high_phase_2_len, dec_phase_2_len, high_phase_3_len) -> Optional[Pattern]:

    buy_price = given_prices[0]
    predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]


  # var predicted_prices = [
  #   {
  #     min: buy_price,
  #     max: buy_price,
  #   },
  #   {
  #     min: buy_price,
  #     max: buy_price,
  #   },
  # ];

    # High Phase 1
    for i in range(2, 2+high_phase_1_len):
        min_pred = math.floor(0.9 * buy_price)
        max_pred = math.ceil(1.4 * buy_price)
        if given_prices[i] is not None:
            if (given_prices[i] < min_pred or given_prices[i] > max_pred):
                # Given price is out of predicted range, so this is the wrong pattern
                return
            # min_pred = given_prices[i]
            # max_pred = given_prices[i]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

    # Dec Phase 1
    min_rate = 6000
    max_rate = 8000
    for i in range(2 + high_phase_1_len, 2 + high_phase_1_len + dec_phase_1_len):
        min_pred = math.floor(min_rate * buy_price / 10000)
        max_pred = math.ceil(max_rate * buy_price / 10000)

        if given_prices[i] is not None:
            if (given_prices[i] < min_pred or given_prices[i] > max_pred):
                # Given price is out of predicted range, so this is the wrong pattern
                return
            # min_pred = given_prices[i]
            # max_pred = given_prices[i]
            min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
            max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))
        min_rate -= 1000
        max_rate -= 400

    # High Phase 2
    for i in range(2 + high_phase_1_len + dec_phase_1_len, 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len):
        min_pred = math.floor(0.9 * buy_price)
        max_pred = math.ceil(1.4 * buy_price)
        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return
            # min_pred = given_prices[i]
            # max_pred = given_prices[i]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

    # Dec Phase 2
    min_rate = 6000
    max_rate = 8000
    for i in range(2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len, 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len):
        min_pred = math.floor(min_rate * buy_price / 10000)
        max_pred = math.ceil(max_rate * buy_price / 10000)
        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return
            # min_pred = given_prices[i]
            # max_pred = given_prices[i]
            min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
            max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

        min_rate -= 1000
        max_rate -= 400

    # High Phase 3
    if 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len + high_phase_3_len != 14:
        raise InconsistentPhaseLengths("Phase lengths don't add up")

    for i in range(2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len, 14):
        min_pred = math.floor(0.9 * buy_price)
        max_pred = math.ceil(1.4 * buy_price)
        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return
            # min_pred = given_prices[i]
            # max_pred = given_prices[i]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

    return Pattern(description="high, decreasing, high, decreasing, high",
                   number=0,
                   prices=predicted_prices)


    # # for (var i = 2; i < 2 + high_phase_1_len; i++) {
  # #   min_pred = Math.floor(0.9 * buy_price);
  # #   max_pred = Math.ceil(1.4 * buy_price);
  # #   if (!isNaN(given_prices[i])) {
  # #     if (given_prices[i] < min_pred || given_prices[i] > max_pred ) {
  # #       // Given price is out of predicted range, so this is the wrong pattern
  # #       return;
  # #     }
  # #     min_pred = given_prices[i];
  # #     max_pred = given_prices[i];
  # #   }
  #
  #   predicted_prices.push({
  #     min: min_pred,
  #     max: max_pred,
  #   });
  # }

  # // Dec Phase 1
  # var min_rate = 6000;
  # var max_rate = 8000;
  # for (var i = 2 + high_phase_1_len; i < 2 + high_phase_1_len + dec_phase_1_len; i++) {
  #   min_pred = Math.floor(min_rate * buy_price / 10000);
  #   max_pred = Math.ceil(max_rate * buy_price / 10000);
  #
  #
  #   if (!isNaN(given_prices[i])) {
  #     if (given_prices[i] < min_pred || given_prices[i] > max_pred ) {
  #       // Given price is out of predicted range, so this is the wrong pattern
  #       return;
  #     }
  #     min_pred = given_prices[i];
  #     max_pred = given_prices[i];
  #     min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price);
  #     max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price);
  #   }
  #
  #   predicted_prices.push({
  #     min: min_pred,
  #     max: max_pred,
  #   });
  #
  #   min_rate -= 1000;
  #   max_rate -= 400;
  # }

  # // High Phase 2
  # for (var i = 2 + high_phase_1_len + dec_phase_1_len; i < 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len; i++) {
  #   min_pred = Math.floor(0.9 * buy_price);
  #   max_pred = Math.ceil(1.4 * buy_price);
  #   if (!isNaN(given_prices[i])) {
  #     if (given_prices[i] < min_pred || given_prices[i] > max_pred ) {
  #       // Given price is out of predicted range, so this is the wrong pattern
  #       return;
  #     }
  #     min_pred = given_prices[i];
  #     max_pred = given_prices[i];
  #   }
  #
  #   predicted_prices.push({
  #     min: min_pred,
  #     max: max_pred,
  #   });
  # }

  # // Dec Phase 2
  # var min_rate = 6000;
  # var max_rate = 8000;
  # for (var i = 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len; i < 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len; i++) {
  #   min_pred = Math.floor(min_rate * buy_price / 10000);
  #   max_pred = Math.ceil(max_rate * buy_price / 10000);
  #
  #
  #   if (!isNaN(given_prices[i])) {
  #     if (given_prices[i] < min_pred || given_prices[i] > max_pred ) {
  #       // Given price is out of predicted range, so this is the wrong pattern
  #       return;
  #     }
  #     min_pred = given_prices[i];
  #     max_pred = given_prices[i];
  #     min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price);
  #     max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price);
  #   }
  #
  #   predicted_prices.push({
  #     min: min_pred,
  #     max: max_pred,
  #   });
  #
  #   min_rate -= 1000;
  #   max_rate -= 400;
  # }

  # // High Phase 3
  # if (2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len + high_phase_3_len != 14) {
  #   throw new Error("Phase lengths don't add up");
  # }
  # for (var i = 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len; i < 14; i++) {
  #   min_pred = Math.floor(0.9 * buy_price);
  #   max_pred = Math.ceil(1.4 * buy_price);
  #   if (!isNaN(given_prices[i])) {
  #     if (given_prices[i] < min_pred || given_prices[i] > max_pred ) {
  #       // Given price is out of predicted range, so this is the wrong pattern
  #       return;
  #     }
  #     min_pred = given_prices[i];
  #     max_pred = given_prices[i];
  #   }
  #
  #   predicted_prices.push({
  #     min: min_pred,
  #     max: max_pred,
  #   });
  # }
  # yield {
  #   pattern_description: "high, decreasing, high, decreasing, high",
  #   pattern_number: 0,
  #   prices: predicted_prices
  # };
# }


def generate_pattern_0(given_prices: List[int]) -> List[Pattern]:

    patterns = []
    for dec_phase_1_len in range(2, 4):  # dec_phase_1_len
        for high_phase_1_len in range(0, 7):  # high_phase_1_len
            for high_phase_3_len in range(0, (7 - high_phase_1_len - 1 + 1)):
                pattern = generate_pattern_0_with_lengths(given_prices, high_phase_1_len, dec_phase_1_len, 7 - high_phase_1_len - high_phase_3_len, 5 - dec_phase_1_len, high_phase_3_len)
                if pattern is not None:
                    patterns.append(pattern)

    return patterns


def generate_pattern_1_with_peak(given_prices: List[int], peak_start: int) -> Optional[Pattern]:

    buy_price = given_prices[0]
    predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]

    min_rate = 8500
    max_rate = 9000

    for i in range(2, peak_start):
        min_pred = math.floor(min_rate * buy_price / 10000)
        max_pred = math.ceil(max_rate * buy_price / 10000)

        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return

            # min_pred = given_prices[i]
            # max_pred = given_prices[i]
            min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
            max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))
        min_rate -= 500
        max_rate -= 300

    #  Now each day is independent of next
    min_randoms = [0.9, 1.4, 2.0, 1.4, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    max_randoms = [1.4, 2.0, 6.0, 2.0, 1.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    for i in range(peak_start, 14):
        min_pred = math.floor(min_randoms[i - peak_start] * buy_price)
        max_pred = math.ceil(max_randoms[i - peak_start] * buy_price)

        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return

            # min_pred = given_prices[i]
            # max_pred = given_prices[i]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

    return Pattern(description="decreasing, high spike, random lows",
                   number=1,
                   prices=predicted_prices)


def generate_pattern_1(given_prices: List[int]) -> List[Pattern]:

    patterns = []
    for peak_start in range(3, 10):
        pattern = generate_pattern_1_with_peak(given_prices, peak_start)

        if pattern is not None:
            patterns.append(pattern)

    return patterns


def generate_pattern_2(given_prices: List[int]) -> List[Pattern]:

    buy_price = given_prices[0]
    predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]

    min_rate = 8500
    max_rate = 9000

    for i in range(2, 14):
        min_pred = math.floor(min_rate * buy_price / 10000)
        max_pred = math.ceil(max_rate * buy_price / 10000)

        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return []

            # min_pred = given_prices[i]
            # max_pred = given_prices[i]
            min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
            max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

        min_rate -= 500
        max_rate -= 300

    pattern = Pattern(description="always decreasing",
                      number=2,
                      prices=predicted_prices)
    return [pattern]


def generate_pattern_3_with_peak(given_prices: List[int], peak_start: int) -> Optional[Pattern]:
    # PATTERN 3: decreasing, spike, decreasing

    buy_price = given_prices[0]
    predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]

    min_rate = 4000
    max_rate = 9000

    for i in range(2, peak_start):
        min_pred = math.floor(min_rate * buy_price / 10000)
        max_pred = math.ceil(max_rate * buy_price / 10000)

        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return

            # min_pred = given_prices[i]
            # max_pred = given_prices[i]
            min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
            max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))
        min_rate -= 500
        max_rate -= 300

    # The Peak
    for i in range(peak_start, peak_start+2):
        min_pred = math.floor(0.9 * buy_price)
        max_pred = math.ceil(1.4 * buy_price)

        if given_prices[i] is not None:
            if given_prices[i] < min_pred or given_prices[i] > max_pred:
                # Given price is out of predicted range, so this is the wrong pattern
                return

            # min_pred = given_prices[i]
            # max_pred = given_prices[i]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

    # Main spike 1
    min_pred = math.floor(1.4 * buy_price) - 1
    max_pred = math.ceil(2.0 * buy_price) - 1
    if given_prices[peak_start+2] is not None:
        if given_prices[peak_start+2] < min_pred or given_prices[peak_start+2] > max_pred:
            # Given price is out of predicted range, so this is the wrong pattern
            return

        # min_pred = given_prices[peak_start+2]
        # max_pred = given_prices[peak_start+2]

    predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[peak_start + 2]))

    # Main spike 2

    min_pred = predicted_prices[peak_start + 2].min
    max_pred = math.ceil(2.0 * buy_price)

    if given_prices[peak_start+3] is not None:
        if given_prices[peak_start+3] < min_pred or given_prices[peak_start+3] > max_pred:
            # Given price is out of predicted range, so this is the wrong pattern
            return

        # min_pred = given_prices[peak_start+3]
        # max_pred = given_prices[peak_start+3]

    predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[peak_start+3]))

    # Main spike 3

    min_pred = math.floor(1.4 * buy_price) - 1
    max_pred = predicted_prices[peak_start + 3].max - 1

    if given_prices[peak_start + 4] is not None:
        if given_prices[peak_start + 4] < min_pred or given_prices[peak_start + 4] > max_pred:
            # Given price is out of predicted range, so this is the wrong pattern
            return

        # min_pred = given_prices[peak_start + 4]
        # max_pred = given_prices[peak_start + 4]

    predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[peak_start + 4]))

    # The rest of the week
    if peak_start + 5 < 14:

        min_rate = 4000
        max_rate = 9000
        for i in range(peak_start+5, 14):
            min_pred = math.floor(min_rate * buy_price / 10000)
            max_pred = math.ceil(max_rate * buy_price / 10000)

            if given_prices[i] is not None:
                if given_prices[i] < min_pred or given_prices[i] > max_pred:
                    # Given price is out of predicted range, so this is the wrong pattern
                    return

                # min_pred = given_prices[i]
                # max_pred = given_prices[i]
                min_rate = minimum_rate_from_given_and_base(given_prices[i], buy_price)
                max_rate = maximum_rate_from_given_and_base(given_prices[i], buy_price)

            predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

            min_rate -= 500
            max_rate -= 300

    return Pattern(description="decreasing, spike, decreasing",
                   number=3,
                   prices=predicted_prices)


def generate_pattern_3(given_prices: List[int]) -> List[Pattern]:
    patterns = []
    for peak_start in range(2, 10):
        pattern = generate_pattern_3_with_peak(given_prices, peak_start)
        if pattern is not None:
            patterns.append(pattern)
    return patterns


def generate_possibilities(sell_prices: List) -> List[Pattern]:
    possibile_patterns = []
    if sell_prices[0] is not None:
        possibile_patterns.extend(generate_pattern_0(sell_prices))
        possibile_patterns.extend(generate_pattern_1(sell_prices))
        possibile_patterns.extend(generate_pattern_2(sell_prices))
        possibile_patterns.extend(generate_pattern_3(sell_prices))
    else:
        for buy_price in range(90, 110):
            sell_prices[0] = sell_prices[1] = buy_price
            possibile_patterns.extend(generate_pattern_0(sell_prices))
            possibile_patterns.extend(generate_pattern_1(sell_prices))
            possibile_patterns.extend(generate_pattern_2(sell_prices))
            possibile_patterns.extend(generate_pattern_3(sell_prices))
    return possibile_patterns


def analyze_possibilities(sell_prices: List) -> Tuple[List[Pattern], Optional[Pattern]]:
    generated_possibilities = generate_possibilities(sell_prices)

    if len(generated_possibilities) == 0:
        return [], None

    global_min_max = []
    for day in range(0, 14):
        if day > 1:
            prices = MinMaxPrice(_min=999, _max=0, actual=generated_possibilities[0].prices[day].actual)

            for poss in generated_possibilities:
                if poss.prices[day].min < prices.min:
                    prices.min = poss.prices[day].min

                if poss.prices[day].max > prices.max:
                    prices.max = poss.prices[day].max
        else:
            prices = MinMaxPrice(_min=generated_possibilities[0].prices[day].min, _max=generated_possibilities[0].prices[day].min, actual=generated_possibilities[0].prices[day].min)

        global_min_max.append(prices)

    min_max_pattern = Pattern(description="predicted min/max across all patterns",
                              number=4,
                              prices=global_min_max)


    # for poss in generated_possibilities:
    #     weekMins = []
    #     weekMaxes = []
    #
    #     for day in poss.prices[1:]:
    #         weekMins.append(day.min)
    #         weekMaxes.append(day.max)
    #
    #     poss.weekMin = min(weekMins)
    #     poss.weekMax = max(weekMaxes)

    def sort_func(possibility: Pattern):
        return possibility.weekMax

    generated_possibilities.sort(reverse=True, key=sort_func)

    return generated_possibilities, min_max_pattern


def fix_sell_prices_length(sell_prices: List[int]):

    for i in range(len(sell_prices), 14):
        sell_prices.append(None)

    return sell_prices


def get_test_predictions() -> Tuple[List[Pattern], Optional[Pattern]]:
    buy_price = 90
    sell_price = [buy_price, buy_price]

    sell_price.append(78)
    sell_price.append(74)
    sell_price.append(None)  # 70)
    sell_price.append(104)

    sell_price = fix_sell_prices_length(sell_price)

    return analyze_possibilities(sell_price)


def get_predictions(prices: List[Prices]) -> Tuple[List[Pattern], Optional[Pattern]]:
    """From a list of buy price and sell prices, predict all possible outcomes from the stalk market."""
    input_prices = []
    for i in range(0, 14):
        if i == 0 or i == 1:
            buy_price = get(prices, day_segment=0)
            input_prices.append(buy_price.price if buy_price is not None else None)
        else:
            sell_price = get(prices, day_segment=i)
            input_prices.append(sell_price.price if sell_price is not None else None)

    return analyze_possibilities(input_prices)


# @dataclass
# class FilledMaxMinPlot:
#     """When plottting these, they must be plotted in the order defined for the fill to work properly."""
#     max_plot: go.Scatter
#     min_plot: go.Scatter
#     avg_plot: go.Scatter
#
#
#     def add_to_fig(self, fig: go.Figure) -> go.Figure:
#         """Helper function to ensure that the plots are added in the correct order. """
#         fig.add_trace(self.max_plot)
#         fig.add_trace(self.min_plot)
#         fig.add_trace(self.avg_plot)
#
#
#     def set_color(self, color):
#         self.max_plot.line.color = color
#         self.min_plot.line.color = color
#         self.avg_plot.line.color = color
#
#     def set_hover_template(self, hovertemplate, custom_text = None):
#
#         self.max_plot.hovertemplate = hovertemplate
#         self.min_plot.hovertemplate = hovertemplate
#         self.avg_plot.hovertemplate = hovertemplate
#
#         if custom_text is not None:
#             self.avg_plot.text = custom_text
#             self.max_plot.text = custom_text
#             self.min_plot.text = custom_text


# def get_filled_scatter_plot(name: str, x_axis: List[Any], mins: List[int], maxs: List[int], avgs: Optional[List[float]] = None) -> FilledMaxMinPlot:
#
#     if avgs is None:
#         avgs = [(_y1 + _y2) / 2 for _y1, _y2 in zip(mins, maxs)]
#
#     x_length = len(mins) if len(mins) > len(maxs) else len(maxs)
#     x = x_axis[:x_length]
#
#     max_plot = go.Scatter(x=x, y=maxs,
#                           mode='lines',
#                           name=f"{name} Max",
#                           line_width=0,
#                           line_shape='spline',
#                           hoverinfo="none",
#                           showlegend=False,
#                           legendgroup=name,
#                           )
#
#     min_plot = go.Scatter(x=x, y=mins,
#                           mode='lines',
#                           fill='tonexty',
#                           name=f"{name} Min",
#                           line_width=0,
#                           line_shape='spline',
#                           hoverinfo="none",
#                           showlegend=False,
#                           legendgroup=name,
#                           )
#
#     avg_plot = go.Scatter(x=x, y=avgs,
#                           mode='lines+markers',
#                           name=name,
#                           line_shape='spline',
#                           legendgroup=name,
#                           )
#
#     return FilledMaxMinPlot(max_plot, min_plot, avg_plot)


# def graph_predictions(user: discord.Member, predictions: List[Pattern], min_max_pattern: Pattern):
#     """Graph the predictions"""
#
#     x_axis = day_segment_names[2:]
#     abs_min_points = [price.min for price in min_max_pattern.prices][2:]
#     abs_max_points = [price.max for price in min_max_pattern.prices][2:]
#
#     avg_points = [0 for i in abs_max_points]
#
#     if min_max_pattern.prices[0].min is not None:
#         buy_price_points = [min_max_pattern.prices[0].min for i in abs_max_points]
#     else:
#         buy_price_points = None
#
#     actual_price_points = [price.actual if price.is_actual_price() else None for price in min_max_pattern.prices][2:]
#
#     for pred in predictions:
#         for i, price in enumerate(pred.prices[2:]):
#             avg_points[i] += price.min + price.max
#
#     avg_points = [i/(len(predictions)*2) for i in avg_points]
#
#     title = f"{user.display_name}'s Stalk Market Predictions" if user is not None else f"Stalk Market Predictions"
#
#     fig: go.Figure = go.Figure(layout_title_text=title,
#                                layout_template="plotly_dark",
#                                layout_xaxis_title="Day of the Week",
#                                layout_yaxis_title="Bells",
#                                )
#
#     plot = get_filled_scatter_plot("Potential Turnip Prices", x_axis, abs_min_points, abs_max_points, avgs=avg_points, )
#     plot.set_color(DEFAULT_PLOTLY_COLORS[0])
#
#     ht = '<b>%{x}</b><br><br>' + \
#          '%{text}' + \
#          '<extra></extra>'
#
#     custom_text = []
#     for i in range(len(abs_min_points)):
#         txt = f"<i>Avg Price</i>: {avg_points[i]:.2f}<br>" +\
#               f"Max Price: {abs_max_points[i]}<br>" + \
#               f"Min Price: {abs_min_points[i]}<br>"
#
#         if actual_price_points[i] is not None:
#             txt += f"Actual Price: {actual_price_points[i]}<br>"
#         if buy_price_points is not None:
#             txt += f"Buy Price: {buy_price_points[i]}<br>"
#
#         custom_text.append(txt)
#
#
#     plot.set_hover_template(ht, custom_text)
#     plot.add_to_fig(fig)
#
#     if buy_price_points is not None:
#         # Add plot indicating the buy price.
#         fig.add_trace(go.Scatter(x=x_axis, y=buy_price_points,
#                                  mode='lines',
#                                  name=f"Buy Price",
#                                  line_dash='dash',
#                                  hoverinfo="none",
#                                  # hovertemplate=ht,
#                                  # text=custom_text,
#                                  # line_width=0,
#                                  # line_shape='spline',
#                                  # showlegend=False,
#                                  # legendgroup=name,
#                                  )
#                       )
#
#     # Add plot indicating the actual price.
#     fig.add_trace(go.Scatter(x=x_axis, y=actual_price_points,
#                              mode='lines',
#                              name=f"Actual Sell Price",
#                              line_dash='dash',
#                              hoverinfo="none",
#                              line_shape='spline',
#                              # hovertemplate=ht,
#                              # text=custom_text,
#                              )
#                   )
#
#     fig.show()
#
#
# def test_graph():
#     # from plotly.colors import DEFAULT_PLOTLY_COLORS
#     # fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
#     # fig.write_html('first_figure.html', auto_open=True)
#     # fig.show()
#
#     title = "Hibiki's Stalk Market Predictions for 4/7/2020"
#     # x = [0,1,2,3,4,5,6,7,8,9]
#     # day_segment_names = ["Sunday Buy Price", "N/A", "Monday Morning", "Monday Afternoon", "Tuesday Morning",
#     #                      "Tuesday Afternoon", "Wednesday Morning", "Wednesday Afternoon", "Thursday Morning",
#     #                      "Thursday Afternoon", "Friday Morning", "Friday Afternoon", "Saturday Morning",
#     #                      "Saturday Afternoon"]
#     # x_rev = day_segment_names[::-1]
#
#     y = [0,2,1,7,2,3,9,2,1,5]
#     # y2 = [1,2,3,2,5,6,7,8,9,0]
#     y2 = [i-3 for i in y]
#     # y_avg = [(_y1 + _y2)/2 for _y1, _y2 in zip(y, y2)]
#     #
#     # y2 = y2[::-1]
#
#     z = [5,7,6,12,7,9,2,6,3,9]
#     # y2 = [1,2,3,2,5,6,7,8,9,0]
#     z2 = [i-3 for i in z]
#     # z_avg = [(_y1 + _y2)/2 for _y1, _y2 in zip(z, z2)]
#
#
#     # x = day_segment_names[:len(y)]
#
#     # x = list(range(len(y)))
#     # x_rev = x[::-1]
#
#     fig: go.Figure = go.Figure(layout_title_text=title,
#                                layout_template="plotly_dark",
#                                layout_xaxis_title="Day of the Week",
#                                layout_yaxis_title="Bells",
#                                )
#     # fig.layout.title.test = title
#     # fig = go.Figure(layout={'title': {'text': title}})
#     # fig = go.Figure(layout=dict(title=dict(text=title)))
#
#     # hovertemplate = '<br><b>X</b>: %{x}<br>' +\
#     #                 '<i>Average Price</i>: %{y:.2f}' +\
#     #                 '<b>%{text}</b>' +\
#     #                 '<extra></extra>'
#
#     hovertemplate = '<b>%{x}</b><br><br>' +\
#                     '<i>Avg Price</i>: %{y:.2f}<br>' +\
#                     '%{text}' + \
#                     '<extra></extra>'
#
#     # text = ['Custom text {}'.format(i + 1) for i in range(len(y))]
#
#     custom_text = []
#     for i in range(len(y)):
#         txt = f"Max Price: {y[i]}<br>" +\
#               f"Min Price: {y2[i]}<br>"
#         custom_text.append(txt)
#
#     # [f'Custom text {i+1}' for i in range(len(y))]
#     # min_text = [f'Custom text {i-1}' for i in range(len(y))]
#
#
#     y_max_min_plot = get_filled_scatter_plot("Y Plot", day_segment_names, y2, y)
#     # y_max_min_plot.avg_plot.hoverinfo = "none"
#     # y_max_min_plot.avg_plot.hoverinfo = "text"
#
#     y_max_min_plot.avg_plot.hovertemplate = hovertemplate
#     y_max_min_plot.avg_plot.text = custom_text
#     # y_max_min_plot.avg_plot.min_text = min_text
#
#     y_max_min_plot.set_color(DEFAULT_PLOTLY_COLORS[0])
#     y_max_min_plot.add_to_fig(fig)
#     #
#     # z_max_min_plot = get_filled_scatter_plot("Z Plot", day_segment_names, z2, z)
#     # z_max_min_plot.set_color(DEFAULT_PLOTLY_COLORS[1])
#     # z_max_min_plot.add_to_fig(fig)
#
#
#     # annotations = []
#     # annotations.append()
#
#
#     fig.show()
#     # fig.show(renderer="svg")


# def filtered_smooth_plot(x_data: List[Any], y_data: List[float]):
#     points: int = 300
#     filttered_y_data = [i if i is not None else 0 for i in y_data]
#
#     import scipy as sp
#     numeric_x = [i for i in range(len(filttered_y_data))]
#
#     x = np.array(numeric_x)
#     y = np.array(filttered_y_data)
#     # noinspection PyArgumentList
#     new_x = np.linspace(x.min(), x.max(), points)
#     new_y = sp.interpolate.interp1d(x, y, kind='quadratic')(new_x)
#
#     # new_y = sp.interpolate.interp1d(x, y, kind='quadratic')(new_x)
#     new_y = [i if i is not None else 0 for i in y_data]
#     return new_x, new_y


def smooth_plot(x_data: List[Any], y_data: List[float]):
    points: int = 30

    # xnew = np.linspace(0, len(x_data), points)
    numeric_x = [i for i in range(len(x_data))]


    # spl = make_interp_spline(numeric_x, y_data, k=3)  # type: BSpline
    # power_smooth = spl(xnew)
    # # return xnew, power_smooth
    #
    # from scipy.interpolate import make_lsq_spline, BSpline
    # t = [-1, 0, 1]
    # k = 3
    # t = np.r_[(numeric_x[0],) * (k + 1),
    #           t,
    #           (numeric_x[-1],) * (k + 1)]
    # spl = make_lsq_spline(numeric_x, y_data, t, k)
    # power_smooth = spl(xnew)
    # return xnew, power_smooth

    import scipy as sp

    x = np.array(numeric_x)
    y = np.array(y_data)
    # noinspection PyArgumentList
    new_x = np.linspace(x.min(), x.max(), points)
    new_y = sp.interpolate.interp1d(x, y, kind='linear')(new_x)

    newer_x = np.linspace(new_x.min(), new_x.max(), points*10)

    newer_y = sp.interpolate.interp1d(new_x, new_y, kind='quadratic')(newer_x)
    # new_y = sp.interpolate.interp1d(x, y, kind='quadratic')(new_x)

    return newer_x, newer_y

    #
    # numeric_x = [i for i in range(len(x_data))]
    # bspl = splrep(xnew, y_data, s=5)
    # # bspl = splprep(numeric_x, y_data, s=5)
    # bspl_y = splev(numeric_x, bspl)
    # return xnew, bspl_y

    # from scipy.interpolate import interp1d
    # # x = np.linspace(0, 10, num=11, endpoint=True)
    # # y = np.cos(-x ** 2 / 9.0)
    # # f = interp1d(x, y)
    # f2 = interp1d(xnew, y_data, kind='cubic')
    # return xnew, f2(xnew)
    #
    # from scipy import signal
    # sy = signal.savgol_filter(y_data, 5, 3)
    # return xnew, sy


# splprep


def matplotgraph_predictions(user: discord.Member, predictions: List[Pattern], min_max_pattern: Pattern, testing=False) -> BytesIO:
    """Graph the predictions"""

    x_axis = day_segment_names[2:]
    abs_min_points = [price.min for price in min_max_pattern.prices][2:]
    abs_max_points = [price.max for price in min_max_pattern.prices][2:]

    avg_points = [0 for i in abs_max_points]

    if min_max_pattern.prices[0].min is not None:
        buy_price_points = [min_max_pattern.prices[0].min for i in abs_max_points]
    else:
        buy_price_points = None

    actual_price_points = [price.actual if price.is_actual_price() else None for price in min_max_pattern.prices][2:]

    for pred in predictions:
        for i, price in enumerate(pred.prices[2:]):
            avg_points[i] += price.min + price.max

    avg_points = [i/(len(predictions)*2) for i in avg_points]

    title = f"{user.display_name}'s Stalk Market Predictions" if user is not None else f"Stalk Market Predictions"

    # Set up the plots

    plt.style.use('dark_background')

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    ax.plot(*smooth_plot(x_axis, avg_points), color="#1f77b4", label="Potential Price")
    ax.plot(x_axis, abs_min_points, color="#000000", alpha=0)
    ax.plot(x_axis, abs_max_points, color="#000000", alpha=0)

    smooth_x, smooth_min_points = smooth_plot(x_axis, abs_min_points)
    smooth_x, smooth_msx_points = smooth_plot(x_axis, abs_max_points)

    ax.fill_between(smooth_x, smooth_min_points, smooth_msx_points, alpha=0.5, color="#1f77b4")

    # ax.plot(x_axis, avg_points)
    # ax.plot(x_axis, abs_min_points)
    # ax.plot(x_axis, abs_max_points)
    if buy_price_points is not None:
        ax.plot(x_axis, buy_price_points,  color="#FF7F0E", alpha=0.7, marker=0, linestyle='None', label="Buy Price")

    ax.plot(x_axis, actual_price_points, 'o', color="#C5FFFF", label="Actual Price")#color="#BD9467")

    legend = ax.legend(shadow=True, fontsize='x-large')

    plt.xticks(np.arange(12), x_axis, rotation=90)  # Set the x ticks to the day names
    # plt.show()

    if testing:
        plt.show()
        return

    imgBuffer = BytesIO()

    plt.savefig(imgBuffer, format="png")
    plt.close()
    return imgBuffer

    """
    fig: go.Figure = go.Figure(layout_title_text=title,
                               layout_template="plotly_dark",
                               layout_xaxis_title="Day of the Week",
                               layout_yaxis_title="Bells",
                               )

    plot = get_filled_scatter_plot("Potential Turnip Prices", x_axis, abs_min_points, abs_max_points, avgs=avg_points, )
    plot.set_color(DEFAULT_PLOTLY_COLORS[0])

    ht = '<b>%{x}</b><br><br>' + \
         '%{text}' + \
         '<extra></extra>'

    custom_text = []
    for i in range(len(abs_min_points)):
        txt = f"<i>Avg Price</i>: {avg_points[i]:.2f}<br>" +\
              f"Max Price: {abs_max_points[i]}<br>" + \
              f"Min Price: {abs_min_points[i]}<br>"

        if actual_price_points[i] is not None:
            txt += f"Actual Price: {actual_price_points[i]}<br>"
        if buy_price_points is not None:
            txt += f"Buy Price: {buy_price_points[i]}<br>"

        custom_text.append(txt)


    plot.set_hover_template(ht, custom_text)
    plot.add_to_fig(fig)

    if buy_price_points is not None:
        # Add plot indicating the buy price.
        fig.add_trace(go.Scatter(x=x_axis, y=buy_price_points,
                                 mode='lines',
                                 name=f"Buy Price",
                                 line_dash='dash',
                                 hoverinfo="none",
                                 # hovertemplate=ht,
                                 # text=custom_text,
                                 # line_width=0,
                                 # line_shape='spline',
                                 # showlegend=False,
                                 # legendgroup=name,
                                 )
                      )

    # Add plot indicating the actual price.
    fig.add_trace(go.Scatter(x=x_axis, y=actual_price_points,
                             mode='lines',
                             name=f"Actual Sell Price",
                             line_dash='dash',
                             hoverinfo="none",
                             line_shape='spline',
                             # hovertemplate=ht,
                             # text=custom_text,
                             )
                  )

    fig.show()
"""

#colors = [’#e6194b’, ‘#3cb44b’, ‘#ffe119’, ‘#4363d8’, ‘#f58231’, ‘#911eb4’, ‘#46f0f0’, ‘#f032e6’, ‘#bcf60c’, ‘#fabebe’, ‘#008080’, ‘#e6beff’, ‘#9a6324’, ‘#fffac8’, ‘#800000’, ‘#aaffc3’, ‘#808000’, ‘#ffd8b1’, ‘#000075’, ‘#808080’, ‘#ffffff’, ‘#000000’]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

    # test_graph()
    buy_price = 90
    sell_price = [buy_price, buy_price]

    sell_price.append(78)
    sell_price.append(74)

    sell_price.append(70)
    sell_price.append(104)

    sell_price.append(167)
    sell_price.append(518)
    #
    sell_price.append(160)
    sell_price.append(98)

    sell_price = fix_sell_prices_length(sell_price)

    possibilities, min_max_pattern = analyze_possibilities(sell_price)

    for prediction in possibilities:
        # desc.append(prediction.description)

       log.info(f"\nDesc: {prediction.description}\n\n"
                f"Sunday Sell:  {prediction.prices[0]}\n"
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
                f"\n")

    # graph_predictions(None, possibilities, min_max_pattern)
    matplotgraph_predictions(None, possibilities, min_max_pattern, testing=True)



    print("Done")



