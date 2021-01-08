#!/usr/bin/env python3

# Library of plotting functions

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator


def plotHeatMap(df, title=None, cmap=None, ax=None, zrange=None, format=".3f"):
    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        cmap = "viridis"
    if zrange is None:
        cmesh = ax.pcolormesh(df, cmap=cmap)
    else:
        cmesh = ax.pcolormesh(df, cmap=cmap, vmin=zrange[0], vmax=zrange[1])

    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns)
    ax.tick_params(direction='in', length=0, pad=10)
    for y in range(df.shape[0]):
        for x in range(df.shape[1]):
            # if df.iloc[y,x]  0:
            ax.text(
                x + 0.5, y + 0.5, '{0:{fmt}}'.format(df.iloc[y, x], fmt=format),
                color="black", fontsize=9, horizontalalignment='center',
                verticalalignment='center', bbox={
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.2,
                    'pad': 0
                })
    ax.set_title(title, fontsize=16)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    cbar = plt.colorbar(cmesh, ax=ax, pad=0.02)
    cbar.ax.tick_params(direction='out', length=3, pad=5)
    return (ax, cbar)


def getColorList(cmap, n):
    cmap = cm.get_cmap(cmap, n)
    colors = []
    for i in range(cmap.N):
        c = matplotlib.colors.to_hex(cmap(i), keep_alpha=True)
        colors.append(c)
    return colors


def rotateXticks(ax, angle):
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)


def rotateYticks(ax, angle):
    for tick in ax.get_yticklabels():
        tick.set_rotation(angle)


def testColorMap(cmap):
    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 1.5 * np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt((X + Y) * (X + Y + 2) * 0.3)

    fig, ax = plt.subplots(figsize=(2, 1.6))
    im = ax.imshow(Z, aspect="auto", interpolation='nearest', vmin=0, origin='lower',
                   cmap=cmap)
    fig.colorbar(im, ax=ax)
    # plt.show()


# Plot grid on the axis ax
def drawGrid(ax, xstep=None, ystep=None, minor_ticks_x=None, minor_ticks_y=None):
    # ax.set_xlim(0, None)
    # ax.set_ylim(0, None)
    if xstep is not None:
        minorLocatorX = MultipleLocator(xstep / minor_ticks_x)
        majorLocatorX = MultipleLocator(xstep)
        ax.xaxis.set_major_locator(majorLocatorX)
        ax.xaxis.set_minor_locator(minorLocatorX)
    if ystep is not None:
        minorLocatorY = MultipleLocator(ystep / minor_ticks_y)
        majorLocatorY = MultipleLocator(ystep)
        ax.yaxis.set_minor_locator(minorLocatorY)
        ax.yaxis.set_major_locator(majorLocatorY)
    ax.minorticks_on()
    ax.grid(which='major', ls=":", lw=.5, alpha=0.8, color="grey")
    ax.grid(which='minor', ls=':', lw=.2, alpha=.8, color='grey')


# Colors
prediction_plot_colors = {
    "targetCNN": "tab:blue",
    "proxyApp": "#92cde3",
    "delta": "#7bba45",
    "predictions": "tab:red",
    "nonGPU": "#b47c98",
    "GPU": "#549f76"
}


# Plot prediction and errors
# DF must have columns:
# 'batch', 'time', 'machine',
# 'ver' - framework name and version
# evaluator,
# 'CPU time predicted', 'GPU time predicted', 'H',
# 'AE', 'APE'
# save - filename to save the plot
def plotPredictionsAndErrors(df, train_set, MAE, MAPE, evaluator, ver=None, model='VGG16',
                             mbs_range=None, save=None):
    colors = prediction_plot_colors
    # Test if df includes necessary columns
    nscolumns = [
        'batch', 'time', 'machine', evaluator, 'GPU time predicted', 'H', 'AE', 'APE'
    ]
    columns = df.columns
    for col in nscolumns:
        if col not in columns:
            print("ERROR: Dataframe df must include column {}".format(col))
            return

    machine = df.iloc[0]['machine']

    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True, dpi=144)
    ax = axs[0]

    if mbs_range is not None:
        left, right = mbs_range
        df = df[df["batch"] <= right]
        df = df[df["batch"] >= left]
        train_set = train_set[train_set["batch"] <= right]
        train_set = train_set[train_set["batch"] >= left]

    train_set.plot("batch", evaluator, lw=0, marker="o", ms=8, color=colors["targetCNN"],
                   label="Used samples", ax=ax)

    df.plot("batch", "time", label="Target CNN epoch time", c=colors["targetCNN"], lw=0,
            marker="o", ms=6, mfc="white", ax=ax)

    if "CPU time predicted" in df.columns:
        df.plot("batch", "CPU time predicted", c=colors["nonGPU"], lw=1, ls=":",
                label="non-GPU time predictions", ax=ax)

    df.plot("batch", "GPU time predicted", c=colors["GPU"], lw=1, ls=":",
            label="GPU time predictions", ax=ax)

    df.plot("batch", "H", c=colors["predictions"], marker="o", lw=0, ms=4, mew=0.8,
            mec='white', label="predictions", ax=ax)

    ax.set_ylabel("time (s)")
    ax.set_xlabel("")
    ax.set_title("{} {} epoch time predictions on {}".format(ver, model, machine))
    ax.legend(fontsize=10, frameon=False, bbox_to_anchor=(1, 1))
    if mbs_range is not None:
        if right - left < 100:
            drawGrid(ax, xstep=10, minor_ticks_x=10)
    else:
        drawGrid(ax, minor_ticks_y=5)

    ax = axs[1]
    ax.set_ylabel("s")
    ax.set_ylim(0, 35)
    ax1 = ax.twinx()
    df.plot.area("batch", "APE", color="coral", lw=1, alpha=0.6, label="APE (%)", ax=ax1)

    df.plot("batch", "AE", lw=1, ls="-", color="black", label="AE (s)", ax=ax)

    ax1.set_ylabel("%")
    ax1.set_ylim(0, 35)
    if mbs_range is not None:
        if right - left < 100:
            drawGrid(ax, xstep=10, minor_ticks_x=10, ystep=10, minor_ticks_y=5)
    else:
        drawGrid(ax, ystep=10, minor_ticks_y=5)

    ax.legend(fontsize=10, loc='upper left', frameon=False, bbox_to_anchor=(1.05, 1))
    ax1.legend(fontsize=10, loc='upper left', frameon=False, bbox_to_anchor=(1.05, .8))
    # MAPE and MAE
    ax1.text(1.06, 0.3, "MAPE {:.2f}%\nMAE  {:.2f}s".format(MAPE, MAE),
             transform=ax1.transAxes)
    ax.set_xlabel("mini-batch size")
    ax.set_xlim(0, None)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
        print("Saved plot to", save)

    plt.show()
    # return fig, ax
