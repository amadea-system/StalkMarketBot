"""
Logic for Stalk Market Predictions
Part of Stalk Market Bot.

Thanks to https://github.com/mikebryant/ac-nh-turnip-prices for the prediction code
"""

PROFILING = False

if PROFILING:
    from cProfile import Profile
    prof = Profile()

    prof.disable()  # i.e. don't time imports

import math
import time
import logging
# import operator
import functools

from itertools import accumulate, chain
from typing import Optional, Dict, List, Union, Tuple #, NamedTuple, Any, TYPE_CHECKING

# import discord
from discord.utils import get

import numpy as np
# from scipy.stats import norm as sci_norm
from scipy import stats as sc_stats
from scipy import signal

from db import Prices
from cy_src.fastStonks import convolve_updf, rate_range_from_given_and_base, get_price, cjs_round


log = logging.getLogger(__name__)

max_guild_predictions = 3

day_segment_names = ["Sunday Buy Price", "N/A", "Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM", "Wed PM",
                     "Thu AM", "Thu PM", "Fri AM", "Fri PM", "Sat AM", "Sat PM"]


rate_multiplier = 10000  # Separately defined in fastStonks.pyx

RATE_MULTIPLIER = rate_multiplier

# used_decays = set()
decay_cache = {}  # Cache for calculating the convolution of two Uniform PDFs


class PatternDefinitions:
    unknown = -1
    roller_coaster = 0
    huge_spike = 1
    always_decreasing = 2
    small_spike = 3

    descriptions = (
        "Roller Coaster",  # 0
        "Huge Spike",  # 1
        "Always Decreasing",  # 2
        "Small Spike",  # 3
        # "predicted min/max across all patterns",  # 4
    )

    probability_matrix = {
        roller_coaster: {
            roller_coaster: 0.2,
            huge_spike: 0.3,
            always_decreasing: 0.15,
            small_spike: 0.35
        },
        huge_spike: {
            roller_coaster: 0.5,
            huge_spike: 0.05,
            always_decreasing: 0.20,
            small_spike: 0.25
        },
        always_decreasing: {
            roller_coaster: 0.25,
            huge_spike: 0.45,
            always_decreasing: 0.05,
            small_spike: 0.25
        },
        small_spike: {
            roller_coaster: 0.45,
            huge_spike: 0.25,
            always_decreasing: 0.15,
            small_spike: 0.15
        },
        unknown: {
            roller_coaster: 0.346278,
            huge_spike: 0.247363,
            always_decreasing: 0.147607,
            small_spike: 0.258752
        },
    }

    @classmethod
    def name(cls, value):
        if value is None or value == -1:
            return "Unknown"
        else:
            return cls.descriptions[value]


pattern_definitions = PatternDefinitions()


class MinMaxPrice:
    __slots__ = ('min', 'max', 'actual')

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
    __slots__ = ('probability', 'prices', 'number', 'description', 'weekMin', 'weekMax', 'guaranteedMin')

    def __init__(self, description: str, number: int, prices: List[MinMaxPrice], probability: float):
        self.probability = probability
        self.prices = prices
        self.number = number
        self.description = description

        self.weekMin = 999
        self.weekMax = 0
        self.guaranteedMin = 999
        self.calculate_min_max()

    def __repr__(self) -> str:
        return f"{self.description} - {self.probability*100}% Chance - Price: {self.weekMin}B<>{self.weekMax}B"


    def calculate_min_max(self):
        # TODO: Optimize

        week_mins = []
        week_maxes = []

        for day in self.prices[2:]:
            week_mins.append(day.min)
            week_maxes.append(day.max)

        # self.weekMin = min((day.min for day in self.prices[2:]))
        # self.weekMax = max((day.max for day in self.prices[2:]))
        # self.guaranteedMin = max((day.min for day in self.prices[2:]))

        self.weekMin = min(week_mins)
        self.weekMax = max(week_maxes)
        self.guaranteedMin = max(week_mins)

    def adjust_probability(self, probability_factor: float):
        self.probability = self.probability * probability_factor


class OverallPatternData:

    def __init__(self, min_max_pattern: Pattern, average_prices: List[float], expected_prices: List[float], total_probabilities: List[float]):
        self.min_max_data = min_max_pattern
        self.average_prices = average_prices
        self.expected_prices = expected_prices
        self.total_probabilities = total_probabilities


class PDF:
    """
    Probability Density Function of rates.
    Since the PDF is continuous*, we approximate it by a discrete probability function:
        the value in range [(x - 0.5), (x + 0.5)) has a uniform probability
        prob[x - value_start];

    Note that we operate all rate on the (* RATE_MULTIPLIER) scale.
    (*): Well not really since it only takes values that "float" can represent in some form, but the
    space is too large to compute directly in JS.
    """

    prob_cache: Dict[Tuple[int, int], np.ndarray] = {}
    decay_cache = {}
    x_cache = {}
    __slots__ = ('value_start', 'value_end', 'use_cache', 'prob', 'prob_length', 'fall_back')

    def __init__(self, a: float, b: float, use_cache=True):  # uniform=True, old_meth=True,
        """
        Initialize a PDF in range [a, b], a and b can be non-integer.
        if uniform is true, then initialize the probability to be uniform, else initialize to a
        all-zero (invalid) PDF.
        """
        self.value_start = _value_start = cjs_round(a)
        self.value_end = _value_end = cjs_round(b)
        total_length = self.value_end - self.value_start
        # self.x = None
        self.use_cache = use_cache
        self.fall_back = False

        # if old_meth:
            # if uniform:

            # for i, val in enumerate(self.prob):  # range(self.value_end - self.value_start + 1):
            #     self.prob[i] = range_intersect_length(self.range_of(i), _range) / total_length

        if (_value_start, _value_end) in self.prob_cache and self.use_cache:
            self.prob = self.prob_cache[(_value_start, _value_end)].copy()  # [:]
        else:
            range1_0 = _value_start - 0.5
            range1_1 = _value_start + 0.5 - 1e-9
            self.prob = np.fromiter(
                (0 if (range1_0 + i > _value_end) or (range1_1 + i < _value_start)
                 else
                 # Range intersect -> (a: max, b: min), range_length -> b - a
                 ((min(range1_1 + i, _value_end) - max(range1_0 + i, _value_start)) / (total_length+1))
                 for i in range(total_length + 1)),
                dtype=float, count=total_length + 1
            )

            if self.use_cache:
                self.prob_cache[(_value_start, _value_end)] = self.prob.copy()  # [:]
        pass

            # else:
            #     self.prob: List[Optional[float]] = [None for i in range(total_length + 1)]
        # else:
        #     if uniform:
        #
        #         if (_value_start, _value_end) in self.prob_cache and self.use_cache:
        #             self.prob = self.prob_cache[(_value_start, _value_end)][:]
        #             # self.x = self.x_cache[(_value_start, _value_end)][:]
        #         else:
        #             # self.x = np.linspace(-1, total_length + 1, total_length + 1)
        #             dist = sc_stats.uniform(0, total_length)
        #             self.prob = dist.pdf(self.x)
        #             if self.use_cache:
        #                 self.prob_cache[(_value_start, _value_end)] = self.prob[:]
        #                 self.x_cache[(_value_start, _value_end)] = self.x[:]
        #     else:
        #         self.prob = np.zeros(total_length + 1)

        # self.prob_length = total_length + 1


    def range_of(self, idx):
        # TODO: consider doing the "exclusive end" properly.
        return (self.value_start + idx - 0.5, self.value_start + idx + 0.5 - 1e-9)


    def min_value(self) -> float:
        return self.value_start - 0.5


    def max_value(self) -> float:
        return self.value_end + 0.5 - 1e-9


    def normalize(self):
        # for i, value in enumerate(self.prob):
        #     self.prob[i] /= total_probability

        # self.prob = [self.prob[i] / total_probability for i in range(len(self.prob))]
        # self.prob = [prob_value / total_probability for prob_value in self.prob]
        total_probability = np.sum(self.prob)
        self.prob = self.prob/total_probability


    def range_limit(self, _range: Tuple[float, float]):
        """
        Limit the values to be in the range, and return the probability that the value was in this
        range.

        """
        start, end = _range
        start = max(start, self.min_value())
        end = min(end, self.max_value())
        if start >= end:
            # Set this to invalid values
            self.value_start = self.value_end = 0
            self.prob = np.array([])
            return 0

        prob = 0
        start_idx = cjs_round(start) - self.value_start
        end_idx = cjs_round(end) - self.value_start

        for i in range(start_idx, end_idx+1):
            bucket_prob = self.prob[i] * range_intersect_length(self.range_of(i), _range)
            self.prob[i] = bucket_prob
            prob += bucket_prob

        # self.prob = np.array(self.prob[start_idx:end_idx + 1])

        self.prob = self.prob[start_idx:end_idx + 1]  # Should already be an ndarray at this point. No need to cast it.

        self.value_start = cjs_round(start)
        self.value_end = cjs_round(end)
        self.normalize()

        return prob


    def decay(self, rate_decay_min, rate_decay_max, old_meth=False):
        """
        Subtract the PDF by a uniform distribution in [rate_decay_min, rate_decay_max]

        For simplicity, we assume that rate_decay_min and rate_decay_max are both integers.


        // O(n^2) naive algorithm for reference, which would be too slow.
        for (let i = this.value_start; i <= this.value_end; i++) {
            const unit_prob = this.prob[i - this.value_start] / (rate_decay_max - rate_decay_min) / 2;
            for (let j = rate_decay_min; j < rate_decay_max; j++) {
                // ([i - 0.5, i + 0.5] uniform) - ([j, j + 1] uniform)
                // -> [i - j - 1.5, i + 0.5 - j] with a triangular PDF
                // -> approximate by
                //    [i - j - 1.5, i - j - 0.5] uniform &
                //    [i - j - 0.5, i - j + 0.5] uniform
                ret.prob[i - j - 1 - ret.value_start] += unit_prob; // Part A
                ret.prob[i - j - ret.value_start] += unit_prob; // Part B
            }
        }
        """
        # rate_decay_min = round_js(rate_decay_min)
        # rate_decay_max = round_js(rate_decay_max)
        rate_decay_min = cjs_round(rate_decay_min)
        rate_decay_max = cjs_round(rate_decay_max)

        if not old_meth:
            # if (rate_decay_min, rate_decay_max) in self.decay_cache and self.use_cache:
            #     decay_pdf = self.decay_cache[(rate_decay_min, rate_decay_max)][:]
            # else:
            #     decay_dist = sc_stats.uniform(0, (rate_decay_max-rate_decay_min))
            #     decay_pdf = decay_dist.pdf(self.x)
            #     if self.use_cache:
            #         self.decay_cache[(rate_decay_min, rate_decay_max)] = decay_pdf[:]

            # decay_pdf = PDF(self.min_value() - rate_decay_max, self.max_value() - rate_decay_min, exper=True)
            # decay_dist = sc_stats.uniform(0, (rate_decay_max-rate_decay_min))
            # decay_pdf = decay_dist.pdf(self.x)
            # self.prob = signal.convolve(decay_pdf, self.prob)[:self.value_end - self.value_start + 1]
            # self.value_start = round_js(self.min_value() - rate_decay_max)
            # self.value_end = round_js(self.max_value() - rate_decay_min)
            # self.prob = fast_convolve(decay_pdf, self.prob, self.value_end - self.value_start + 1)
            # self.normalize()

            # --- Cython Approximation --- #
            # used_decays.add((0, self.value_end-self.value_start, 0, rate_decay_max-rate_decay_min))

            if not self.fall_back:# and (self.value_end-self.value_start) != rate_decay_max-rate_decay_min:
                # self.prob = convolve_updf(0, self.value_end-self.value_start, 0, rate_decay_max-rate_decay_min)
                self.prob, self.fall_back = cached_decay(0, self.value_end-self.value_start, 0, rate_decay_max-rate_decay_min)
                # No need to normalize as it's done in the c function

                self.value_start = cjs_round((self.value_start - 0.5) - rate_decay_max)
                self.value_end = cjs_round((self.value_end + 0.5 - 1e-9) - rate_decay_min)
            else:
                self.fall_back = True
                # log.warning("Decay Fallback")
                decay_dist = sc_stats.uniform(0, (rate_decay_max - rate_decay_min))
                decay_pdf = decay_dist.pdf(
                    range(max(len(self.prob), rate_decay_max - rate_decay_min + 1)))

                self.value_start = cjs_round((self.value_start - 0.5) - rate_decay_max)
                self.value_end = cjs_round((self.value_end + 0.5 - 1e-9) - rate_decay_min)
                self.prob = signal.convolve(decay_pdf, self.prob)[:self.value_end - self.value_start + 1]
                self.normalize()

        else:
            ret = PDF(self.min_value() - rate_decay_max, self.max_value() - rate_decay_min, False)


            # Transform to "CDF"
            # for i in range(1, len(self.prob)):
            #     self.prob[i] += self.prob[i-1]
            self.prob = list(accumulate(self.prob))


        # Return this.prob[l - this.value_start] + ... + this.prob[r - 1 - this.value_start];
        # This assume that this.prob is already transformed to "CDF".

        # @functools.lru_cache(maxsize=256)
        # def pdf_sum(l, r):
        #     l -= self.value_start
        #     r -= self.value_start
        #
        #     if l < 0:
        #         l = 0
        #     if r > len(self.prob):
        #         r = len(self.prob)
        #
        #     if l >= r:
        #         return 0
        #
        #     subtractor = 0 if l == 0 else self.prob[int(l-1)]
        #
        #     return self.prob[int(r - 1)] - subtractor


        # for x, value in enumerate(ret.prob):
        #     #  i - j - 1 - ret.value_start == x  (Part A)
        #     # -> i = x + j + 1 + ret.value_start, j in [rate_decay_min, rate_decay_max)
        #
        #     ret.prob[x] = pdf_sum(x + rate_decay_min + 1 + ret.value_start, x + rate_decay_max + ret.value_start)
        #     ret.prob[x] += pdf_sum(x + rate_decay_min + ret.value_start, x + rate_decay_max + ret.value_start)

        # ret.prob = [pdf_sum(x + rate_decay_min + 1 + ret.value_start, x + rate_decay_max + ret.value_start) +
        #             pdf_sum(x + rate_decay_min + ret.value_start, x + rate_decay_max + ret.value_start) for x, value in
        #             enumerate(ret.prob)]

            ret.prob = fast_pdf_sum(self.prob, self.value_start, rate_decay_min, rate_decay_max, ret.value_start, len(ret.prob))

        # ret.prob = self.pdf_sum(ret, rate_decay_min, rate_decay_max)

            self.prob = ret.prob
            self.value_start = ret.value_start
            self.value_end = ret.value_end

            self.normalize()


class InconsistentPhaseLengths(Exception):
    pass


def cached_decay(a1: int, b1: int, a2: int, b2: int) -> Tuple[np.ndarray, bool]:
    if b1 == b2:
        b1 += 1  # Very hacky, but doesn't effect the probabilities substantially.

    if a1 + b2 < a2 + b1:
        if (a1, b1, a2, b2) in decay_cache:
            return decay_cache[(a1, b1, a2, b2)].copy(), False  #[:]
        else:
            decay = convolve_updf(a1, b1, a2, b2)
            decay_cache[(a1, b1, a2, b2)] = decay.copy()  #[:]
            return decay.copy(), False  #[:], False

    elif a1 + b2 > a2 + b1:  # if a1 + b2 > a2 + b1 then we need to call decay_cache with the arguments flipped.

        if (a2, b2, a1, b1) in decay_cache:
            return decay_cache[(a2, b2, a1, b1)].copy(), False  #[:], False

        else:
            decay = convolve_updf(a2, b2, a1, b1)
            decay_cache[(a2, b2, a1, b1)] = decay.copy()  # [:]
            return decay.copy(), False  #[:], False
    else:
        # raise NotImplementedError
        log.warning(f"Tri Convolving w/ (a1, b1, a2, b2): {(a1, b1, a2, b2)}")
        decay = tri_conv(a1, b1, a2, b2)
        return decay, True


def tri_conv(a1: int, b1: int, a2: int, b2: int):
    a1_p_a2 = a1 + a2  # Start
    b1_p_b2 = b1 + b2  # End
    length = b1_p_b2 - a1_p_a2  # End - Start
    convolution = np.zeros(length + 1, dtype=float)  # full(length +1, 1 / (b1 - a1), dtype=float)
    divisor = ((b1 - a1) * (b2 - a2))
    temp = 0
    _sum = (1 / (b1 - a1)) * ((a2 + b1) - (a1 + b2))


    for x in range(0, a1 + b2):
        if a1_p_a2 <= x:  # and x <  a1 + b2:
            temp = (x - a1_p_a2) / divisor
            convolution[x] = temp
            _sum += temp
        else:  # x < a1_p_a2:
            convolution[x] = 0

    for x in range((a2 + b1), length + 1):

        if x < b1_p_b2:
            temp = ((-x) + b1_p_b2) / divisor
            convolution[x] = temp
            _sum += temp
        else:  # b1_p_b2 <= x
            convolution[x] = 0

    for x in range(0, length+1):
        convolution[x] = convolution[x]/_sum

    return convolution#/sum  #np.sum(convolution)


def test_aprox_convolve(a1: int, b1: int, a2: int, b2: int):
    """Unused. For testing purposes"""

    a1_p_a2 = a1 + a2  # Start
    b1_p_b2 = b1 + b2  # End
    length = b1_p_b2 - a1_p_a2  # End - Start
    convolution = np.zeros(length + 1, dtype=float) #full(length +1, 1 / (b1 - a1), dtype=float)
    divisor = ((b1 - a1) * (b2 - a2))
    temp = 0
    _sum = (1 / (b1 - a1)) * ((a2 + b1) - (a1 + b2))
    # middle_section = 1 / (b1 - a1)
    middle_section = 1 / (b1 - a2)


    for x in range(0, a1 + b2):
        if a1_p_a2 <= x:  # and x <  a1 + b2:
            temp = (x - a1_p_a2) / divisor
            convolution[x] = temp
            _sum += temp
        else:  # x < a1_p_a2:
            convolution[x] = 0

    for x in range((a2 + b1), length + 1):

        if x < b1_p_b2:
            temp = ((-x) + b1_p_b2) / divisor
            convolution[x] = temp
            _sum += temp
        else:  # b1_p_b2 <= x
            convolution[x] = 0

        # middle_section = 1 / (b1 - a1)
    for x in range((a1 + b2), (a2 + b1)):
        convolution[x] = middle_section


    for x in range(0, length+1):
        convolution[x] = convolution[x]/_sum

    return convolution#/sum  #np.sum(convolution)


def fast_pdf_sum(prob: List, value_start: int, rate_decay_min: int, rate_decay_max: int, ret_value_start: int, ret_length: int):
    """Unused. Original convolve aproximation from JS code"""
    rate_decay_min = int(rate_decay_min)
    rate_decay_max = int(rate_decay_max)
    ret_value_start = int(ret_value_start)
    ret_length = int(ret_length)

    output = []
    prob_length = len(prob)
    # ret_value_start = ret_value_start
    # prob = prob
    append_to_output = output.append
    for x in range(ret_length):
        temp = 0
        l = x + rate_decay_min + 1 + ret_value_start
        r = x + rate_decay_max + ret_value_start

        l -= value_start
        r -= value_start

        if l < 0:
            l = 0

        if r > prob_length:
            r = prob_length

        if l >= r:
             temp += 0
        else:

            # subtractor = 0 if l == 0 else self.prob[int(l - 1)]
            # temp += self.prob[int(r - 1)] - subtractor
            subtractor = 0 if l == 0 else prob[l - 1]
            temp += prob[r - 1] - subtractor

        l = x + rate_decay_min + ret_value_start
        r = x + rate_decay_max + ret_value_start

        l -= value_start
        r -= value_start

        if l < 0:
            l = 0

        if r > prob_length:
            r = prob_length

        if l >= r:
            temp += 0
        else:

            # subtractor = 0 if l == 0 else self.prob[int(l - 1)]
            # temp += self.prob[int(r - 1)] - subtractor
            subtractor = 0 if l == 0 else prob[l - 1]
            temp += prob[r - 1] - subtractor

        append_to_output(temp)

    return output


# region Functions Moved to optimizedStonks
# def intceil(value) -> int:
#     """Function that more closely mimics the ceil function in AC NH"""
#     return math.trunc(value + 0.99999)
#
#
# def round_js(value: Union[int, float]) -> int:
#     """Repacement for the PY round function that mimics the JS Math.round function"""
#
#     x = math.floor(value)
#     if (value - x) < .50:
#         return x
#     else:
#         return math.ceil(value)


#
# def minimum_rate_from_given_and_base(given_price: int, _buy_price: int) -> float:
#     return rate_multiplier * (given_price - 0.99999) / _buy_price
#
#
# def maximum_rate_from_given_and_base(given_price: int, _buy_price: int) -> float:
#     return rate_multiplier * (given_price + 0.00001) / _buy_price
#
#
# def rate_range_from_given_and_base(given_price: int, _buy_price: int) -> Tuple[float, float]:
#
#     return (
#         minimum_rate_from_given_and_base(given_price, _buy_price),
#         maximum_rate_from_given_and_base(given_price, _buy_price)
#     )
#
#
# def get_price(_rate, _base_price) -> int:
#     # return intceil(_rate * _base_price / rate_multiplier)
#     return math.trunc(_rate * _base_price / rate_multiplier + 0.99999)
# endregion


def clamp(x: float, min_value: int, max_value: int) -> Union[float, int]: # TODO: Move to optimizedStonks
    return min(max(x, min_value), max_value)


@functools.lru_cache(maxsize=256)
def range_length(_range: Tuple[float, float]) -> float:
    return _range[1] - _range[0]


@functools.lru_cache(maxsize=256)
def range_intersect(range1: Tuple[float, float], range2: Tuple[float, float]) -> Optional[Tuple[float, float]]:

    if range1[0] > range2[1] or range1[1] < range2[0]:
        return None

    return max(range1[0], range2[0]), min(range1[1], range2[1])


@functools.lru_cache(maxsize=256)
def range_intersect_length(range1: Tuple[float, float], range2: Tuple[float, float]) -> float:

    if range1[0] > range2[1] or range1[1] < range2[0]:
        return 0

    return min(range1[1], range2[1]) - max(range1[0], range2[0])

    # return range_length(range_intersect(range1, range2))


class PredictPatterns:

    def __init__(self):
        self.fudge_factor = 0

    def multiply_pattern_probability(self, patterns: List[Pattern], probability: float) -> List[Pattern]:

        # an example of what could call this:
        # multiply_generator_probability( generator = generate_pattern_1_with_peak(given_prices, peak_start),
        #                                 probability = 1 / (10 - 3)

        for pattern in patterns:
            pattern.adjust_probability(probability)

        return patterns


    def generate_individual_random_price(self, given_prices: List[int], predicted_prices: List[MinMaxPrice], start: int, length: int, rate_min: float, rate_max: float) -> float:
        """
            * This corresponds to the code:
            *   for (int i = start; i < start + length; i++)
            *   {
            *     sellPrices[work++] =
            *       intceil(randfloat(rate_min / RATE_MULTIPLIER, rate_max / RATE_MULTIPLIER) * basePrice);
            *   }
            *
            * Would return the conditional probability given the given_prices, and modify
            * the predicted_prices array.
            * If the given_prices won't match, returns 0.
        """
        # FIXME: Actually explicitly return the modified predicted_prices List instead of the whole modify thing we are doing now.

        rate_min *= rate_multiplier
        rate_max *= rate_multiplier

        buy_price = given_prices[0]
        rate_range = (rate_min, rate_max)

        prob = 1

        for i in range(start, start+length):
            min_pred = get_price(rate_min, buy_price)
            max_pred = get_price(rate_max, buy_price)

            if given_prices[i] is not None:
                if given_prices[i] < min_pred - self.fudge_factor or given_prices[i] > max_pred + self.fudge_factor:
                    # Given price is out of predicted range, so this is the wrong pattern
                    return 0  # FIXME: I think this should be None

                # TODO: How to deal with probability when there's fudge factor?
                # Clamp the value to be in range now so the probability won't be totally biased to fudged values.
                real_rate_range = rate_range_from_given_and_base(clamp(given_prices[i], min_pred, max_pred), buy_price)
                prob *= range_intersect_length(rate_range, real_rate_range) / range_length(rate_range)

                if prob == 0:
                    log.warning(f"Prob was 0 in generate_individual_random_price")

                min_pred = given_prices[i]
                max_pred = given_prices[i]

            predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

        return prob


    def generate_decreasing_random_price(self, given_prices: List[int], predicted_prices: List[MinMaxPrice], start: int, length: int, start_rate_min, start_rate_max, rate_decay_min, rate_decay_max):
        # region Description
        """
         This corresponds to the code:
           rate = randfloat(start_rate_min, start_rate_max);
           for (int i = start; i < start + length; i++)
           {
             sellPrices[work++] = intceil(rate * basePrice);
             rate -= randfloat(rate_decay_min, rate_decay_max);
           }

         Would return the conditional probability given the given_prices, and modify
         the predicted_prices array.
         If the given_prices won't match, returns 0.
          """
        # endregion

        # pdf_archive = []
        start_rate_min *= RATE_MULTIPLIER
        start_rate_max *= RATE_MULTIPLIER
        rate_decay_min *= RATE_MULTIPLIER
        rate_decay_max *= RATE_MULTIPLIER

        buy_price = given_prices[0]
        rate_pdf = PDF(start_rate_min, start_rate_max)
        prob = 1

        for i in range(start, start + length):
            # min_pred = get_price(rate_pdf.min_value(), buy_price)
            # max_pred = get_price(rate_pdf.max_value(), buy_price)
            min_pred = get_price(rate_pdf.min_value(),  buy_price)
            max_pred = get_price(rate_pdf.max_value(), buy_price)

            if given_prices[i] is not None:
                if given_prices[i] < min_pred - self.fudge_factor or given_prices[i] > max_pred + self.fudge_factor:
                    # Given price is out of predicted range, so this is the wrong pattern
                    return 0  # FIXME: I think this should be None

                # TODO: How to deal with probability when there's fudge factor?
                # Clamp the value to be in range now so the probability won't be totally biased to fudged values.
                real_rate_range = rate_range_from_given_and_base(clamp(given_prices[i], min_pred, max_pred), buy_price)
                prob *= rate_pdf.range_limit(real_rate_range)
                if prob == 0:
                    return 0

                min_pred = given_prices[i]
                max_pred = given_prices[i]

            predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[i]))

            # pdf_archive.append(rate_pdf.prob[:])

            rate_pdf.decay(rate_decay_min, rate_decay_max)
            # log.info("")

        return prob


    def generate_peak_price(self, given_prices: List[int], predicted_prices: List[MinMaxPrice], start: int, rate_min: float, rate_max: float):
        """
         This corresponds to the code:
           rate = randfloat(rate_min, rate_max);
           sellPrices[work++] = intceil(randfloat(rate_min, rate) * basePrice) - 1;
           sellPrices[work++] = intceil(rate * basePrice);
           sellPrices[work++] = intceil(randfloat(rate_min, rate) * basePrice) - 1;

         Would return the conditional probability given the given_prices, and modify
         the predicted_prices array.
         If the given_prices won't match, returns 0.
        """

        rate_min *= RATE_MULTIPLIER
        rate_max *= RATE_MULTIPLIER

        buy_price = given_prices[0]
        prob = 1
        rate_range = (rate_min, rate_max)

        # Calculate the probability first.
        # Prob(middle_price)
        middle_price = given_prices[start + 1]
        if middle_price is not None:
            min_pred = get_price(rate_min, buy_price)
            max_pred = get_price(rate_max, buy_price)
            if middle_price < min_pred - self.fudge_factor or middle_price > max_pred + self.fudge_factor:
                # Given price is out of predicted range, so this is the wrong pattern
                return 0  # FIXME: I think this should be None

            # TODO: How to deal with probability when there's fudge factor?
            # Clamp the value to be in range now so the probability won't be totally biased to fudged values.
            real_rate_range = rate_range_from_given_and_base(clamp(middle_price, min_pred, max_pred), buy_price)
            prob *= range_intersect_length(rate_range, real_rate_range) / range_length(rate_range)

            if prob == 0:
                return 0

            rate_range = range_intersect(rate_range, real_rate_range)

        left_price = given_prices[start]
        right_price = given_prices[start + 2]

        """
         Prob(left_price | middle_price), Prob(right_price | middle_price)
        
         A = rate_range[0], B = rate_range[1], C = rate_min, X = rate, Y = randfloat(rate_min, rate)
         rate = randfloat(A, B); sellPrices[work++] = intceil(randfloat(C, rate) * basePrice) - 1;
        
         => X->U(A,B), Y->U(C,X), Y-C->U(0,X-C), Y-C->U(0,1)*(X-C), Y-C->U(0,1)*U(A-C,B-C),
         let Z=Y-C,  Z1=A-C, Z2=B-C, Z->U(0,1)*U(Z1,Z2)
         Prob(Z<=t) = integral_{x=0}^{1} [min(t/x,Z2)-min(t/x,Z1)]/ (Z2-Z1)
         let F(t, ZZ) = integral_{x=0}^{1} min(t/x, ZZ)
            1. if ZZ < t, then min(t/x, ZZ) = ZZ -> F(t, ZZ) = ZZ
            2. if ZZ >= t, then F(t, ZZ) = integral_{x=0}^{t/ZZ} ZZ + integral_{x=t/ZZ}^{1} t/x
                                         = t - t log(t/ZZ)
         Prob(Z<=t) = (F(t, Z2) - F(t, Z1)) / (Z2 - Z1)
         Prob(Y<=t) = Prob(Z>=t-C)
        """

        for price in (left_price, right_price):
            if price is None:
                continue

            min_pred = get_price(rate_min, buy_price) - 1
            max_pred = get_price(rate_range[1], buy_price) - 1
            if price < min_pred - self.fudge_factor or price > max_pred + self.fudge_factor:
                # Given price is out of predicted range, so this is the wrong pattern
                return 0

            # TODO: How to deal with probability when there's fudge factor?
            # Clamp the value to be in range now so the probability won't be totally biased to fudged values.
            rate2_range = rate_range_from_given_and_base(clamp(price, min_pred, max_pred)+1, buy_price)

            def F(t, ZZ):
                if t <= 0:
                    return 0

                if ZZ < t:
                    return ZZ
                else:
                    return t - t * (math.log(t) - math.log(ZZ))

            a, b = rate_range
            c = rate_min
            z1 = a - c
            z2 = b - c

            def PY(t):
                return (F(t - c, z2) - F(t - c, z1)) / (z2 - z1)

            prob *= PY(rate2_range[1]) - PY(rate2_range[0])

            if prob == 0:
                return 0

        """
         * Then generate the real predicted range.
         We're doing things in different order then how we calculate probability,
         since forward prediction is more useful here.
        
         Main spike 1
        """

        min_pred = get_price(rate_min, buy_price) - 1
        max_pred = get_price(rate_max, buy_price) - 1

        if given_prices[start] is not None:
            min_pred = given_prices[start]
            max_pred = given_prices[start]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[start]))

        # Main Spike 2
        min_pred = predicted_prices[start].min
        max_pred = get_price(rate_max, buy_price)

        if given_prices[start+1] is not None:
            min_pred = given_prices[start+1]
            max_pred = given_prices[start+1]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[start+1]))

        # Main Spike 3
        min_pred = get_price(rate_min, buy_price) - 1
        max_pred = predicted_prices[start + 1].max - 1

        if given_prices[start+2] is not None:
            min_pred = given_prices[start + 2]
            max_pred = given_prices[start + 2]

        predicted_prices.append(MinMaxPrice(min_pred, max_pred, given_prices[start+2]))

        return prob


    def generate_pattern_0_with_lengths(self, given_prices: List[int], high_phase_1_len, dec_phase_1_len, high_phase_2_len,
                                        dec_phase_2_len, high_phase_3_len) -> Optional[Pattern]:

        buy_price = given_prices[0]
        predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]

        probability = 1

        # High Phase 1
        probability *= self.generate_individual_random_price(given_prices, predicted_prices, start=2, length=high_phase_1_len,
                                                        rate_min=0.9, rate_max=1.4)
        if probability == 0:
            return None

        # Dec Phase 1
        probability *= self.generate_decreasing_random_price(given_prices, predicted_prices, start=2 + high_phase_1_len,
                                                        length=dec_phase_1_len, start_rate_min=0.6, start_rate_max=0.8,
                                                        rate_decay_min=0.04, rate_decay_max=0.1)
        if probability == 0:
            return None

        # High Phase 2
        probability *= self.generate_individual_random_price(given_prices, predicted_prices,
                                                        start=2 + high_phase_1_len + dec_phase_1_len,
                                                        length=high_phase_2_len, rate_min=0.9, rate_max=1.4)
        if probability == 0:
            return None

        # Dec Phase 2
        probability *= self.generate_decreasing_random_price(given_prices, predicted_prices,
                                                        start=2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len,
                                                        length=dec_phase_2_len, start_rate_min=0.6, start_rate_max=0.8,
                                                        rate_decay_min=0.04, rate_decay_max=0.1)
        if probability == 0:
            return None

        # High Phase 3
        if 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len + high_phase_3_len != 14:
            raise InconsistentPhaseLengths("Phase lengths don't add up")

        prev_length = 2 + high_phase_1_len + dec_phase_1_len + high_phase_2_len + dec_phase_2_len
        probability *= self.generate_individual_random_price(given_prices, predicted_prices,
                                                        start=prev_length, length=14-prev_length,
                                                        rate_min=0.9, rate_max=1.4)
        if probability == 0:
            return None

        return Pattern(description=pattern_definitions.descriptions[0],  #"Roller Coaster",#"high, decreasing, high, decreasing, high",
                       number=0,
                       prices=predicted_prices,
                       probability=probability)


    def generate_pattern_0(self, given_prices: List[int], adjust_prob=True) -> List[Pattern]:

        patterns = []
        for dec_phase_1_len in range(2, 4):  # dec_phase_1_len
            for high_phase_1_len in range(0, 7):  # high_phase_1_len
                for high_phase_3_len in range(0, (7 - high_phase_1_len - 1 + 1)):
                    pattern = self.generate_pattern_0_with_lengths(given_prices, high_phase_1_len, dec_phase_1_len, 7 - high_phase_1_len - high_phase_3_len, 5 - dec_phase_1_len, high_phase_3_len)
                    if pattern is not None:
                        if adjust_prob:
                            pattern.adjust_probability(1 / (4 - 2) / 7 / (7 - high_phase_1_len))  # TODO: Simplify this
                        patterns.append(pattern)
                    # yield list(pattern)

        return patterns


    def generate_pattern_1_with_peak(self, given_prices: List[int], peak_start: int) -> Optional[Pattern]:

        buy_price = given_prices[0]
        predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]
        probability = 1

        probability *= self.generate_decreasing_random_price(given_prices, predicted_prices, start=2,
                                                        length=peak_start-2, start_rate_min=0.85, start_rate_max=0.9,
                                                        rate_decay_min=0.03, rate_decay_max=0.05)
        if probability == 0:
            return None

        #  Now each day is independent of next
        min_randoms = [0.9, 1.4, 2.0, 1.4, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        max_randoms = [1.4, 2.0, 6.0, 2.0, 1.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        for i in range(peak_start, 14):

            probability *= self.generate_individual_random_price(given_prices, predicted_prices, start=i,
                                                            length=1, rate_min=min_randoms[i - peak_start],
                                                            rate_max=max_randoms[i - peak_start])
            if probability == 0:
                return None

        return Pattern(description=pattern_definitions.descriptions[1],  # "Huge Spike",#"decreasing, high spike, random lows",
                       number=1,
                       prices=predicted_prices,
                       probability=probability)


    def generate_pattern_1(self, given_prices: List[int], adjust_prob=True) -> List[Pattern]:

        patterns = []
        for peak_start in range(3, 10):
            pattern = self.generate_pattern_1_with_peak(given_prices, peak_start)

            if pattern is not None:
                if adjust_prob:
                    pattern.adjust_probability(1/(10-3))  # TODO: Simplify this
                patterns.append(pattern)
        return patterns


    def generate_pattern_2(self, given_prices: List[int]) -> List[Pattern]:

        buy_price = given_prices[0]
        predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]
        probability = 1

        probability *= self.generate_decreasing_random_price(given_prices, predicted_prices, start=2,
                                                        length=14 - 2, start_rate_min=0.85, start_rate_max=0.9,
                                                        rate_decay_min=0.03, rate_decay_max=0.05)
        if probability == 0:
            return []

        pattern = Pattern(description=pattern_definitions.descriptions[2],  #"Always Decreasing",
                          number=2,
                          prices=predicted_prices,
                          probability=probability)
        return [pattern]


    def generate_pattern_3_with_peak(self, given_prices: List[int], peak_start: int) -> Optional[Pattern]:
        # PATTERN 3: decreasing, spike, decreasing

        buy_price = given_prices[0]
        predicted_prices = [MinMaxPrice(buy_price, buy_price, buy_price), MinMaxPrice(buy_price, buy_price, buy_price)]
        probability = 1

        probability *= self.generate_decreasing_random_price(given_prices, predicted_prices, start=2,
                                                        length=peak_start - 2, start_rate_min=0.4, start_rate_max=0.9,
                                                        rate_decay_min=0.03, rate_decay_max=0.05)
        if probability == 0:
            return None

        # The Peak
        probability *= self.generate_individual_random_price(given_prices, predicted_prices, start=peak_start,
                                                        length=2, rate_min=0.9, rate_max=1.4)
        if probability == 0:
            return None

        probability *= self.generate_peak_price(given_prices, predicted_prices, start=peak_start+2, rate_min=1.4, rate_max=2.0)
        if probability == 0:
            return None

        # The rest of the week
        if (peak_start + 5) < 14:
            probability *= self.generate_decreasing_random_price(given_prices, predicted_prices, start=peak_start+5,
                                                            length=14-(peak_start+5), start_rate_min=0.4, start_rate_max=0.9,
                                                            rate_decay_min=0.03, rate_decay_max=0.05)
            if probability == 0:
                return None

        return Pattern(description=pattern_definitions.descriptions[3],  #"Small Spike",#"decreasing, spike, decreasing",
                       number=3,
                       prices=predicted_prices,
                       probability=probability)


    def generate_pattern_3(self, given_prices: List[int], adjust_prob=True) -> List[Pattern]:
        patterns = []
        for peak_start in range(2, 10):
            pattern = self.generate_pattern_3_with_peak(given_prices, peak_start)
            if pattern is not None:
                if adjust_prob:
                    pattern.adjust_probability(1/(10-2))  # TODO: Simplify to constant 0.125
                patterns.append(pattern)
        return patterns


    def get_transition_probability(self, previous_pattern: Optional[int]) -> Dict[int, float]:
        """Gets a lookup table for probability modifiers for a given previous pattern"""
        if previous_pattern is None or previous_pattern == -1:
            return pattern_definitions.probability_matrix[-1]
        else:
            return pattern_definitions.probability_matrix[previous_pattern]


    def generate_all_patterns(self, sell_prices: List[int],  previous_pattern: Optional[int] = -1) -> List[Pattern]:
        """Generates all 4 patterns for a list of prices and a previous pattern"""
        fx_list = (self.generate_pattern_0, self.generate_pattern_1, self.generate_pattern_2, self.generate_pattern_3)
        transition_probability = self.get_transition_probability(previous_pattern)

        patterns = []
        for i, fx in enumerate(fx_list):
            patterns.extend(self.multiply_pattern_probability(fx(sell_prices), transition_probability[i]))
            # patterns.extend(fx(sell_prices))

        return patterns

        # yield from (
        #     multiply_pattern_probability(fx(sell_prices), transition_probability[i])
        #     for
        #     i, fx in enumerate(fx_list)
        # )


    def generate_possibilities(self, sell_prices: List, previous_pattern: Optional[int] = -1, first_buy: bool = False) -> List[Pattern]:
        """Returns a generator of all potential possibilities for the given criteria."""

        if sell_prices[0] is not None and not first_buy:
            return self.generate_all_patterns(sell_prices, previous_pattern)

        else:
            _possibilities = []
            for _buy_price in range(90, 111):
                sell_prices[0] = sell_prices[1] = _buy_price
                if first_buy:
                    _possibilities.extend(self.generate_pattern_3(sell_prices))
                else:
                    _possibilities.extend(self.generate_all_patterns(sell_prices, previous_pattern))

            return _possibilities


def analyze_possibilities(sell_prices: List, previous_pattern: Optional[int] = -1, first_buy: bool = False) -> Tuple[List[Pattern], Optional[OverallPatternData]]:
    """Computes and analyzes the possible patterns and their probabilities."""
    pattern_generator = PredictPatterns()
    generated_possibilities = []

    for i in range(6):  # Iterate over possible fudge factors until we produce a match.
        pattern_generator.fudge_factor = i
        generated_possibilities = pattern_generator.generate_possibilities(sell_prices, previous_pattern, first_buy)
        if len(generated_possibilities) > 0:
            log.info(f"Generated {len(generated_possibilities)} possibility using a fudge factor of {i}")
            break

  #   js_poss = [
  # 2.1504873407530748e-10,
  # 2.1504873407530748e-10,
  # 2.0125357003386677e-10,
  # 2.0125357003386677e-10,
  # 1.8847987405462362e-10,
  # 1.8847987405462362e-10,
  # 1.7664211484085697e-10,
  # 1.7664211484085697e-10,
  # 1.6566273436528217e-10,
  # 1.6566273436528217e-10,
  # 1.5547133051426187e-10,
  # 1.5547133051426187e-10,
  # 1.4600393096068125e-10,
  # 1.4600393096068125e-10,
  # 1.3720234726909888e-10,
  # 1.3720234726909888e-10,
  # 1.2901359965894776e-10,
  # 1.2901359965894776e-10,
  # 6.451560716507736e-11,
  # 6.451560716507736e-11,
  # 1.142857144254127e-40,
  # 1.142857144254127e-40,
  # 1.0772692613951331e-16,
  # 1.0772692613951331e-16,
  # 6.496825101911992e-15,
  # 6.496825101911992e-15,
  # 6.978835868603419e-14,
  # 6.978835868603419e-14,
  # 3.7001301679819326e-13,
  # 3.7001301679819326e-13,
  # 8.728927908585588e-15,
  # 8.728927908585588e-15,
  # 1.3326874052884643e-12,
  # 8.728927908585588e-15,
  # 8.728927908585588e-15,
  # 1.3326874052884643e-12
# ]

#     js_poss = [
#   1.2042729108217221e-8,
#   1.2042729108217221e-8,
#   1.127019992189654e-8,
#   1.127019992189654e-8,
#   1.0554872947058924e-8,
#   1.0554872947058924e-8,
#   9.89195843108799e-9,
#   9.89195843108799e-9,
#   9.277113124455802e-9,
#   9.277113124455802e-9,
#   8.706394508798664e-9,
#   8.706394508798664e-9,
#   8.17622013379815e-9,
#   8.17622013379815e-9,
#   7.683331447069538e-9,
#   7.683331447069538e-9,
#   7.224761580901076e-9,
#   7.224761580901076e-9,
#   3.612874001244332e-9,
#   3.612874001244332e-9,
#   6.400000007823111e-39,
#   6.400000007823111e-39,
#   6.032707863812747e-15,
#   6.032707863812747e-15,
#   3.6382220570707156e-13,
#   3.6382220570707156e-13,
#   3.908148086417915e-12,
#   3.908148086417915e-12,
#   2.0720728940698822e-11,
#   2.0720728940698822e-11,
#   9.77639925761586e-13,
#   9.77639925761586e-13,
#   7.463049469615401e-11,
#   9.77639925761586e-13,
#   9.77639925761586e-13,
#   7.463049469615401e-11
# ]

    if len(generated_possibilities) == 0:
        log.info(f"Unable to find any predictions for: {sell_prices} \n w/ prev pat: {previous_pattern}")
        return [], None

    # for i in range (len(js_poss)):
    #     if js_poss[i] != generated_possibilities[i].probability*100:
    #         print(f"Misimatch: {i}, {(js_poss[i], generated_possibilities[i].probability*100)}")

    """
    for:  
    sell_price = [None, None]
    sell_price.append(90)
    sell_price.append(90)
    sell_price.append(90)
    sell_price.append(90)
    sell_price.append(90)
    sell_price.append(90)
    Misimatch: 30, (8.728927908585588e-15, 2.4803363772483137e-22)
    Misimatch: 31, (8.728927908585588e-15, 1.2378059610220155e-22)
    Misimatch: 33, (8.728927908585588e-15, 1.7552712236684096e-30)
    Misimatch: 34, (8.728927908585588e-15, 8.759639231988134e-31)
    """

    category_prob_totals = [0, 0, 0, 0]
    total_probability = functools.reduce(lambda acc, it: acc + it.probability, generated_possibilities, 0)

    for poss in generated_possibilities:
        poss.probability /= total_probability  # Normalize the probabilities
        category_prob_totals[poss.number] += poss.probability  # Calculate the total probability for each pattern type.


    # for poss in generated_possibilities:
    #
    #     week_mins = []
    #     week_maxes = []
    #     for day in poss.prices[2:]:
    #         if day.min != day.max:
    #             we
    #
    #

    global_min_max = []
    average_prices = []
    expected_prices = [0 for i in range(0, 14)]
    for day in range(0, 14):
        average_price = []
        if day > 1:
            prices = MinMaxPrice(_min=999, _max=0, actual=generated_possibilities[0].prices[day].actual)

            for poss in generated_possibilities:
                if poss.prices[day].min < prices.min:
                    prices.min = poss.prices[day].min

                if poss.prices[day].max > prices.max:
                    prices.max = poss.prices[day].max

                if generated_possibilities[0].prices[day].actual is not None:
                    average_price.append(generated_possibilities[0].prices[day].actual)
                    expected_prices[day] = generated_possibilities[0].prices[day].actual
                else:
                    average_price.append((poss.prices[day].min + poss.prices[day].max)/2)
                    expected_prices[day] += ((poss.prices[day].min + poss.prices[day].max)/2) * poss.probability
        else:
            prices = MinMaxPrice(_min=generated_possibilities[0].prices[day].min, _max=generated_possibilities[0].prices[day].min, actual=generated_possibilities[0].prices[day].min)

        global_min_max.append(prices)
        if len(average_price) > 0:
            average_prices.append(sum(average_price)/len(average_price))

    min_max_pattern = Pattern(description="predicted min/max across all patterns",
                              number=4,
                              prices=global_min_max, probability=0)

    # def sort_func(possibility: Pattern):
    #     return possibility.weekMax

    # Sort by Overall prob, then sub prob
    def sort_func(possibility: Pattern):
        return (category_prob_totals[possibility.number], possibility.probability)  # possibility.weekMax

    generated_possibilities.sort(reverse=True, key=sort_func)

    other_data = OverallPatternData(min_max_pattern, average_prices, expected_prices[2:], category_prob_totals)

    return generated_possibilities, other_data


def fix_sell_prices_length(sell_prices: List[int]):

    for i in range(len(sell_prices), 14):
        sell_prices.append(None)

    return sell_prices


def get_test_predictions() -> Tuple[List[Pattern], Optional[Pattern], Optional[List[float]]]:
    # buy_price = 90
    # sell_price = [buy_price, buy_price]
    #
    # sell_price.append(78)
    # sell_price.append(74)
    # sell_price.append(None)  # 70)
    # sell_price.append(104)

    buy_price = 100
    sell_price = [buy_price, buy_price]

    sell_price.append(90)
    sell_price.append(119)
    # sell_price.append(None)  # 70)
    # sell_price.append(104)

    sell_price = fix_sell_prices_length(sell_price)
    prev_pattern = pattern_definitions.huge_spike

    return analyze_possibilities(sell_price, previous_pattern=prev_pattern)


def get_predictions(prices: List[Prices], previous_pattern: int) -> Tuple[List[Pattern], Optional[OverallPatternData]]:
    """From a list of buy price and sell prices, predict all possible outcomes from the stalk market."""
    input_prices = []
    for i in range(0, 14):
        if i == 0 or i == 1:
            buy_price = get(prices, day_segment=0)
            input_prices.append(buy_price.price if buy_price is not None else None)
        else:
            sell_price = get(prices, day_segment=i)
            input_prices.append(sell_price.price if sell_price is not None else None)

    return analyze_possibilities(input_prices, previous_pattern)


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







#colors = [’#e6194b’, ‘#3cb44b’, ‘#ffe119’, ‘#4363d8’, ‘#f58231’, ‘#911eb4’, ‘#46f0f0’, ‘#f032e6’, ‘#bcf60c’, ‘#fabebe’, ‘#008080’, ‘#e6beff’, ‘#9a6324’, ‘#fffac8’, ‘#800000’, ‘#aaffc3’, ‘#808000’, ‘#ffd8b1’, ‘#000075’, ‘#808080’, ‘#ffffff’, ‘#000000’]




def rollercoaster_testt_prices():
    buy_price = None  # 100
    # buy_price = 90
    sell_price = [buy_price, buy_price]

    sell_price.append(90)
    sell_price.append(90)

    sell_price.append(90)
    sell_price.append(90)

    sell_price.append(90)
    sell_price.append(90)
    return sell_price


def large_spike_test():
    buy_price = 100
    # buy_price = 90
    sell_price = [buy_price, buy_price]

    sell_price.append(90)
    sell_price.append(119)
    return sell_price


def weird_test():
    buy_price = 102
    # buy_price = 90
    sell_price = [buy_price, buy_price]

    sell_price.append(95)
    sell_price.append(105)
    sell_price.append(200)

    return sell_price


def no_findy():
    # return [110, 110, 86, 81, 96, None, 126, None, 182, None, None, None, None, None]
    return [110, 110, 86, 81, 96, None, 126, None, None, None, None, None, None, None]



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

    # test_graph()
    # buy_price = 90
    # sell_price = [buy_price, buy_price]
    #
    # sell_price.append(78)
    # sell_price.append(74)
    #
    # sell_price.append(70)
    # sell_price.append(104)
    #
    # sell_price.append(167)
    # sell_price.append(518)
    # #
    # sell_price.append(160)
    # sell_price.append(98)

    buy_price = None#100
    # buy_price = 90
    sell_price = [buy_price, buy_price]


    # sell_price.append(109)
    # sell_price.append(None)  # 70)
    # sell_price.append(104)
    sell_price = weird_test() #no_findy() #large_spike_test()#w
    sell_price = fix_sell_prices_length(sell_price)
    prev_pattern = 2#2 #pattern_definitions.huge_spike  # 1

    sell_price = fix_sell_prices_length(sell_price)

    if PROFILING:
        prof.enable()  # profiling back on

    start = time.perf_counter()
    possibilities, other_data = analyze_possibilities(sell_price, prev_pattern)
    log.info(f"Took: {time.perf_counter()-start} s")

    if PROFILING:
        prof.disable()  # don't profile the generation of stats
        prof.dump_stats('mystats-4-27.pstat')

    # test_graph()
    #
    if len(possibilities) > 0:
        for i, prob in enumerate(other_data.total_probabilities):
            log.info(f"{prob*100}% chance of a {pattern_definitions.name(i)} pattern")

    # for prediction in possibilities:
    #     # desc.append(prediction.description)
    #    log.info(f"\nDesc: {prediction.description}\n\n"
    #             f"Sunday Sell:  {prediction.prices[0]}\n"
    #             f"Monday AM:    {prediction.prices[2]}\n"
    #             f"Monday PM:    {prediction.prices[3]}\n"
    #             f"Tuesday AM:   {prediction.prices[4]}\n"
    #             f"Tuesday PM:   {prediction.prices[5]}\n"
    #             f"Wednesday AM: {prediction.prices[6]}\n"
    #             f"Wednesday AM: {prediction.prices[7]}\n"
    #             f"Thursday AM:  {prediction.prices[8]}\n"
    #             f"Thursday AM:  {prediction.prices[9]}\n"
    #             f"Friday AM:    {prediction.prices[10]}\n"
    #             f"Friday AM:    {prediction.prices[11]}\n"
    #             f"Saturday AM:  {prediction.prices[12]}\n"
    #             f"Saturday AM:  {prediction.prices[13]}"
    #             f"\n")

    # graph_predictions(None, possibilities, min_max_pattern)
    # matplotgraph_predictions(None, possibilities, min_max_pattern, testing=True)

    # stats = perf_stats.stats()

    # range_intersect_length 4173
    # Range_intersect        998
    # Range length           538

    # Generate decreating with price 18678
    # log.info(f"Calls to Decay: \n{used_decays}")
    print("Done")



