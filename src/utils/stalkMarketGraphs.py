"""
Graphing code for Stalk Market Predictions
Part of Stalk Market Bot.
"""

import logging

from io import BytesIO
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple, NamedTuple, Any

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
# from scipy.ndimage.filters import gaussian_filter1d
# from scipy.interpolate import splrep, splev, splprep
import numpy as np

import discord

from utils.stalkMarketPredictions import day_segment_names, Pattern, fix_sell_prices_length, analyze_possibilities

if TYPE_CHECKING:
    from cogs.stalkMarket import UserPredictions

log = logging.getLogger(__name__)


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

    # noinspection PyArgumentList
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

def matplotgraph_predictions(user: discord.Member, predictions: List[Pattern], min_max_pattern: Pattern, average_prices: List[float], testing=False) -> BytesIO:
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

    # for pred in predictions:
    #     for i, price in enumerate(pred.prices[2:]):
    #         avg_points[i] += price.min + price.max

    # avg_points = [i/(len(predictions)*2) for i in avg_points]
    avg_points = average_prices

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
        return None

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

def matplotgraph_guild_predictions(users_predictions: List['UserPredictions']) -> BytesIO:
    """Graph the predictions"""

    max_graphs = 3

    x_axis = day_segment_names[2:]

    plt.style.use('dark_background')

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    for i, pred in enumerate(users_predictions):
        if i >= max_graphs:
            break

        best_price_points = [price.actual if price.is_actual_price() else price.max for price in pred.best().prices][2:]
        ax.plot(*smooth_plot(x_axis, best_price_points), label=f"{pred.user_name} - Best")

        # avg_price_points = pred.average
        # ax.plot(*smooth_plot(x_axis, avg_price_points), label=f"{pred.user_name} - Average")

    legend = ax.legend(shadow=True, fontsize='medium')
    plt.xticks(np.arange(12), x_axis, rotation=90)  # Set the x ticks to the day names

    # plt.show()

    imgBuffer = BytesIO()

    plt.savefig(imgBuffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    return imgBuffer


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
    #
    # sell_price = fix_sell_prices_length(sell_price)
    #
    # possibilities, min_max_pattern, avg_prices = analyze_possibilities(sell_price)
    #
    # for prediction in possibilities:
    #     # desc.append(prediction.description)
    #
    #     log.info(f"\nDesc: {prediction.description}\n\n"
    #              f"Sunday Sell:  {prediction.prices[0]}\n"
    #              f"Monday AM:    {prediction.prices[2]}\n"
    #              f"Monday PM:    {prediction.prices[3]}\n"
    #              f"Tuesday AM:   {prediction.prices[4]}\n"
    #              f"Tuesday PM:   {prediction.prices[5]}\n"
    #              f"Wednesday AM: {prediction.prices[6]}\n"
    #              f"Wednesday AM: {prediction.prices[7]}\n"
    #              f"Thursday AM:  {prediction.prices[8]}\n"
    #              f"Thursday AM:  {prediction.prices[9]}\n"
    #              f"Friday AM:    {prediction.prices[10]}\n"
    #              f"Friday AM:    {prediction.prices[11]}\n"
    #              f"Saturday AM:  {prediction.prices[12]}\n"
    #              f"Saturday AM:  {prediction.prices[13]}"
    #              f"\n")
    #
    # # graph_predictions(None, possibilities, min_max_pattern)
    # # matplotgraph_predictions(None, possibilities, min_max_pattern, testing=True)
    #
    # print("Done")
