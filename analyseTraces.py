#!/usr/bin/env python3

# Unified script based on Jupyter Notebooks
# aggregateTimeFromMultipleTraces-PytorchGooglenetV100.ipynb
# aggregateTimeFromMultipleTraces-PytorchGooglenetV100-overlappingKernels.ipynb
# aggregateTimeFromMultipleTraces-PytorchGooglenetP100-overlappingKernels

import os
import argparse
import pandas as pd
import subprocess
import re
import sys
import string
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import seaborn as sns
import numpy as np
from cycler import cycler
import itertools
from lib import plotter
import yaml
import argparse

version = "1.02a"
print("Analyser of series of nsys traces. v.{}.".format(version))
# Default parameters

target = "pytorch Googlenet V100"
targetdir = "traces_analysed/{}".format(''.join(target.split(" ")))
logdir = 'logs/p3.2xlarge.cont/traces/20210108'
trace_name_pattern = 'nsys_trace_([0-9]+).json'
event_patterns = ['Iteration [123456789]+']
dataiodir = "../mlbench/pytorch/logs/p3.2xlarge.cont/iterseries/resnet18_3_20210224/"
time_logs = "../mlbench/pytorch/logs/p3.2xlarge.cont/batchseries/imagenet/pytorch1.4.0_googlenet_20201130/clean_logs.csv"
dnnmark_logs = "../mlbench/dnnmark/logs/p3.2xlarge.cont/dnnmark_pytorch_googlenet_ConvSeries_20201130/CNN_time.csv"

convert_traces = False

parser = argparse.ArgumentParser()
parser.add_argument("--conf", "-c", help="YAML file with parameters")
parser.add_argument("--reparse", action="store_true", default=False)
args = parser.parse_args()

convert_traces = args.reparse
if args.conf is not None:
    print("Reading config file")
    print(args.conf)
    if os.path.isfile(args.conf):
        with open(args.conf) as f:
            conf = yaml.safe_load(f)
        target = conf["target"]
        targetdir = "traces_analysed/{}".format(''.join(target.split(" ")))
        logdir = conf["logdir"]
        trace_name_pattern = conf["trace_name_pattern"]
        event_patterns = conf["event_patterns"]
        dataiodir = conf["dataiodir"]
        time_logs = conf["time_logs"]
        dnnmark_logs = conf["dnnmark_logs"]

print("Parameters:")
print(target)
print(targetdir)
print(logdir)
print(trace_name_pattern)
print(event_patterns)
print(dataiodir)
print(time_logs)
print(dnnmark_logs)


def filterMemoryOps(df):
    mask5a = (df['name'].str.contains('cudaMemcpyAsync'))
    mask5b = (df['name'].str.contains('cudaStreamSynchronize'))
    mask5c = (df['name'].str.contains('cudaMemsetAsync'))
    mask5d = (df['name'].str.contains('cudaEvent'))
    #     mask5e = (df['name'].str.contains('cudaHostAlloc'))
    df = df.loc[~(mask5a | mask5b | mask5c | mask5d)].copy()

    return df


def filerCudaEvent(df):
    mask1 = (df['name'].str.contains('cudaEvent'))
    df = df.loc[~(mask1)].copy()
    return df


# Get int number from string
def parseIteration(s):
    return int(s.strip(string.ascii_letters).strip(' ,'))


# Calculate gaps for each iteration
def calculateGaps(df, batch='param', iteration='iteration'):
    callsandgaps = None
    df_ = df.copy()
    if 'prev' in df_.columns:
        df_.drop('prev', axis=1, inplace=True)
    if 'gap' in df_.columns:
        df_.drop('gap', axis=1, inplace=True)
    for mbs, mbsdf in df_.groupby(batch, sort=True):
        for it, iterdf in mbsdf.groupby(iteration, sort=False):
            iterdf = iterdf.sort_values(['start']).reset_index(drop=True)
            iterdf.loc[:, 'prev'] = iterdf['end'].shift(1)
            iterdf.loc[:, 'gap'] = iterdf['start'] - iterdf['prev']
            if callsandgaps is None:
                callsandgaps = iterdf
            else:
                callsandgaps = pd.concat([callsandgaps, iterdf],
                                         ignore_index=True)
    return callsandgaps


# Return DF with events (names) from the list of async event
def selectEventsByName(df, names):
    asyncEventsdf = None
    for aname in names:
        adf = df[df['name'].str.contains(aname)]
        if asyncEventsdf is None:
            asyncEventsdf = adf.copy()
        else:
            asyncEventsdf = pd.concat([asyncEventsdf, adf], ignore_index=False)
    return asyncEventsdf


# Create target dir
if not os.path.exists(targetdir):
    os.makedirs(targetdir)
print("Target dir", targetdir)

# Read trace files
list_command = "ls -1 " + logdir
files = []
param_values = []
proc = subprocess.Popen(list_command.split(" "),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding='utf8')
for line in iter(proc.stdout.readline, ''):
    line = line.strip(" \n")
    m = re.match(trace_name_pattern, line)
    if m:
        files.append(os.path.abspath(os.path.join(logdir, line)))
        param_values.append(m.group(1))

print('{} files in {}'.format(len(files), logdir))

results = None
for param, tracefile in zip(param_values, files):
    events = ' '.join(event_patterns)
    if convert_traces:
        # Run
        # python3 parseOneTrace.py -f $tracefile --events $events
        command = 'python3 parseOneTrace.py -f {} --events {}'.format(
            tracefile, events)
        print(command)
        p = subprocess.run(command.split(' '),
                           stdin=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           bufsize=0,
                           shell=False)
        if p.returncode == 0:
            print('Finished OK')
        else:
            print(p.stdout.decode('utf-8'))
            print('ERROR')
            print(p.stderr.decode('utf-8'))
    # Read data from CSV file
    directory = os.path.dirname(tracefile)
    csvfile = ('').join(os.path.basename(tracefile).split('.')
                        [:-1])  # Filename without extension
    csvfile = csvfile + '.csv'
    csvfile = os.path.join(directory, csvfile)
    print('Reading {}'.format(csvfile))
    df_ = pd.read_csv(csvfile)
    df_['param'] = param
    if results is None:
        results = df_
    else:
        results = results.append(df_, ignore_index=True)

results[['start', 'end', 'duration']] = results[['start', 'end',
                                                 'duration']].astype(float)
results[['param']] = results[['param']].astype(int)

# 2.2 Parse iteration numbers
longtimes = results.copy()

# PArse Iteration numbers
longtimes['NVTX'] = longtimes['NVTX'].fillna('')
longtimes.loc[:, 'iteration'] = longtimes["NVTX"].apply(parseIteration)
longtimes.drop(['NVTX'], axis=1, inplace=True)
longtimes[['param', 'corrID', 'Type']] = longtimes[['param', 'corrID',
                                                    'Type']].astype(int)
iterations = list(longtimes['iteration'].unique())
longtimes = longtimes[longtimes['iteration'].isin(iterations[2:-2])]
print("Iterations:", sorted(longtimes['iteration'].unique()))
print("mbs:", sorted(longtimes['param'].unique()))

# 2.3 Gap times
# Aggregate duration and gaps per mbs and per iteration
df_ = longtimes[(longtimes['GPU side'] == False)
                & (longtimes['corrID'] != 0)].copy()
# # No memory operations
# df_ = filterMemoryOps(df_)
cputime = pd.DataFrame(columns=[
    'batch', 'iteration', 'mean call time', 'mean gap time',
    'median call time', 'median gap time', 'sum calls', 'sum gaps', 'calls'
],
                       dtype=float)
callsandgaps = None
for mbs, mbsdf in df_.groupby('param', sort=True):
    for iteration, iterdf in mbsdf.groupby('iteration', sort=False):
        iterdf = iterdf.sort_values(['start']).reset_index(drop=True)
        iterdf.loc[:, 'prev'] = iterdf['end'].shift(1)
        iterdf.loc[:, 'gap'] = iterdf['start'] - iterdf['prev']
        if callsandgaps is None:
            callsandgaps = iterdf
        else:
            callsandgaps = pd.concat([callsandgaps, iterdf], ignore_index=True)
        meancall = iterdf['duration'].mean()
        meangap = iterdf['gap'].mean()
        mediancall = iterdf['duration'].median()
        mediangap = iterdf['gap'].median()
        sumcalls = iterdf['duration'].sum()
        sumgaps = iterdf['gap'].sum()
        calls = iterdf.shape[0]
        cputime.loc[cputime.shape[0], :] = [
            mbs, iteration, meancall, meangap, mediancall, mediangap, sumcalls,
            sumgaps, calls
        ]

callsandgaps.drop(["GPU side", "Type"], axis=1, inplace=True)
cputime[['batch', 'iteration',
         'calls']] = cputime[['batch', 'iteration', 'calls']].astype(int)

# 2.5 Remove overlapping events with names from the list
df_ = callsandgaps.copy()
async_names = [
    'cudaEventQuery', 'cudaMemsetAsync', 'cudaEventDestroy', 'cudaEventRecord',
    'cudaBindTexture', 'cudaUnbindTexture', 'cudaEventCreateWithFlags',
    'cudaHostAlloc'
]
asynceventsdf = None

while df_[(df_['gap'] < 0)].shape[0] > 0:
    overlapping = (df_[(df_['gap'] < 0)].shape[0])
    total = df_.shape[0]
    print("Overlaps: {}/{}".format(overlapping, total))
    overlapping_events = df_[(df_['gap'].shift(-1) < 0) |
                             (df_['gap'] < 0)].copy()
    print("Names of overlapping events:\n",
          overlapping_events['name'].unique())
    overlapping_events = selectEventsByName(overlapping_events, async_names)
    print("Filtered overlapping events:", overlapping_events['name'].unique())
    if asynceventsdf is None:
        asynceventsdf = overlapping_events
    else:
        asynceventsdf = pd.concat([asynceventsdf, overlapping_events])
    df_ = df_[~df_.index.isin(overlapping_events.index)]
    df_ = calculateGaps(df_)

nooverlapdf = df_.copy()

# Start and end of each iteration
# Gaps between iterations
# Get start iteration time
df_iter = nooverlapdf.copy()
df_iter_g = df_iter.groupby(['param', 'iteration']).agg({
    'start': 'min',
    'end': 'max'
})
df_iter_g = df_iter_g.reset_index(drop=False).rename(
    {
        'start': 'iter_start',
        'end': 'iter_end'
    }, axis=1)
df_iter_g.loc[:, 'prev'] = df_iter_g.shift(1)['iter_end']
df_iter_g.loc[:, 'iter_gap'] = df_iter_g['iter_start'] - df_iter_g['prev']
df_iter_g = df_iter_g[df_iter_g['iter_gap'] >= 0]

meanitergaps = df_iter_g.groupby(['param']).agg({'iter_gap': 'mean'})

fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
df_iter_g.plot.scatter(x="param", y="iter_gap", alpha=0.7, ax=ax)
ax.set_title("Gaps between iterations\n{}".format(logdir))
ax.set_ylabel("gaps (s)")
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
meanitergaps.plot(ax=ax)
figfile = "gapsBetweenIterations.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

df_iter = pd.merge(df_iter, df_iter_g, on=["param", "iteration"], how="outer")
df_iter.loc[:, 'initer_start'] = df_iter['start'] - df_iter['iter_start']

# Time of CPU calls to compute kernels
# Remove all cuda... not calculation events
cuda_names = [
    'cudaMemcpyAsync', 'cudaStreamSynchronize', 'cudaStreamWaitEvent_v3020',
    'cudaEventCreate', 'cudaEventQuery', 'cudaMemsetAsync', 'cudaEventDestroy',
    'cudaEventRecord', 'cudaBindTexture', 'cudaUnbindTexture',
    'cudaEventCreateWithFlags', 'cudaHostAlloc'
]
df_ = callsandgaps.copy()
cuda_events = selectEventsByName(df_, cuda_names)
df_ = df_[~df_.index.isin(cuda_events.index)]

df_.loc[:, 'callandgap'] = df_['duration'] + df_['gap']
df_ = df_.groupby(['param', 'name',
                   'iteration']).agg('sum').reset_index(drop=False)
df_ = df_.groupby(['param',
                   'name']).agg('median').reset_index(drop=False).drop(
                       ['iteration', 'corrID', 'end', 'prev'], axis=1)
clean_cpu = df_.copy()

# 2.6 Plot time per event
# Select memory operations
event = "cudaMemcpyAsync"
syncevents = selectEventsByName(nooverlapdf, [event])

df_ = syncevents.copy()
ignorecolumns = ['start', 'end', 'corrID', 'prev', 'gap']
df_ = df_.drop(ignorecolumns, axis=1)
df_ = df_.groupby(['param', 'name',
                   'iteration']).agg('sum').reset_index(drop=False)

df_ = df_.groupby(['param', 'name']).agg('mean').reset_index(drop=False)
cpucudaMemcpy = df_.copy()
df_['duration'] = df_['duration'] * 1000.
df_ = df_.pivot(index="name", columns=['param'], values=['duration']).fillna(0)
df_.columns = df_.columns.get_level_values(1)

fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
col1 = plotter.getColorList('viridis', n=10)
col2 = plotter.getColorList('tab10', n=10)
# col3 = getColorList('tab20c',n=20)
colors = col2 + col1
N = len(colors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined",
                                                           colors,
                                                           N=N)

df_.T.plot(marker='o', ms=4, mfc='w', ax=ax)
ax.set_xlim(0, None)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title("{} time per iteration for {}".format(event, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
# handles = handles[::-1][:limit]
# labels = labels[::-1][:limit]
leg = ax.legend(handles,
                labels,
                ncol=1,
                loc='upper left',
                frameon=False,
                bbox_to_anchor=(1, 1.05))
figfile = "cudaMemcpyAsync.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Box plot for CPU events and gaps without memory ops
df_ = filterMemoryOps(callsandgaps).copy()
df_['duration'] = df_['duration'] * 1000.
df_['gap'] = df_['gap'] * 1000.
df_ = df_.rename({'param': 'batch'}, axis=1)
fliers = dict(markerfacecolor='tab:blue',
              ms=2,
              mec="tab:blue",
              alpha=0.3,
              marker='o')
axs = df_.boxplot(by='batch',
                  column=['gap', 'duration'],
                  figsize=(15, 8),
                  flierprops=fliers)
for ax in axs:
    ax.grid(ls=":", alpha=0.5)
    ax.set_ylabel('time (ms)', fontsize=18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_title(ax.get_title(), fontsize=20)
plt.suptitle("{}. Calls and gaps (NO memory operations)".format(target),
             fontsize=24)
figfile = "eventsAndGaps_noMemops.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Box plot for CPU events and gaps after removing overlapping events
df_ = nooverlapdf.copy()
df_['duration'] = df_['duration'] * 1000.
df_['gap'] = df_['gap'] * 1000.
df_ = df_.rename({'param': 'batch'}, axis=1)
fliers = dict(markerfacecolor='tab:blue',
              ms=2,
              mec="tab:blue",
              alpha=0.3,
              marker='o')
axs = df_.boxplot(by='batch',
                  column=['gap', 'duration'],
                  figsize=(15, 8),
                  flierprops=fliers)
for ax in axs:
    ax.grid(ls=":", lw=0.7)
    ax.set_ylabel('time (ms)', fontsize=18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_title(ax.get_title(), fontsize=20)
plt.suptitle("{}. Calls and gaps (NO overlapping events)".format(target),
             fontsize=24)
figfile = "eventsAndGaps_noOverlaps.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# No overlaps, no memory (only datacopy events)
memoryops = selectEventsByName(nooverlapdf,
                               ["cudaStreamSynchronize", "cudaMemcpyAsync"])
nomemorydf = nooverlapdf[~nooverlapdf['corrID'].isin(memoryops['corrID'].
                                                     unique())].copy()

# Box plot for CPU events and gaps after removing overlapping events
df_ = nomemorydf.copy()
df_['duration'] = df_['duration'] * 1000.
df_['gap'] = df_['gap'] * 1000.
df_ = df_.rename({'param': 'batch'}, axis=1)
fliers = dict(markerfacecolor='tab:blue',
              ms=2,
              mec="tab:blue",
              alpha=0.3,
              marker='o')
axs = df_.boxplot(by='batch',
                  column=['gap', 'duration'],
                  figsize=(15, 8),
                  flierprops=fliers)
for ax in axs:
    ax.grid(ls=":", lw=0.7)
    ax.set_ylabel('time (ms)', fontsize=18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_title(ax.get_title(), fontsize=20)
plt.suptitle(
    "{}. Calls and gaps (NO overlapping events no datacopy)".format(target),
    fontsize=24)
figfile = "eventsAndGaps_noOverlaps_nodatacopy.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Aggregate events
nomemorydf.loc[:, 'callandgap'] = nomemorydf['duration'] + nomemorydf['gap']
nomemorydf = nomemorydf.groupby(['param', 'name', 'iteration'
                                 ]).agg('sum').reset_index(drop=False)
nomemorydf = nomemorydf.groupby(['param', 'name']).agg('mean').reset_index(
    drop=False).drop(['iteration', 'corrID', 'start', 'end', 'prev'], axis=1)

# Area plot of events and gaps
for val, title, fig_ in [[
        'callandgap', 'CPU calls and gaps', 'CPUcallsAndGapsArea'
], ['duration', 'CPU calls', 'CPUcallsArea'],
                         ['gap', 'CPU gaps', 'CPUgapsArea']]:
    df_ = nomemorydf.copy()
    df_[val] = df_[val] * 1000.
    df_ = df_.pivot(index="name", columns=['param'], values=[val]).fillna(0)
    df_.loc[:, 'total'] = df_.sum(axis=1)
    df_ = df_.sort_values(['total'])
    df_.drop(['total'], axis=1, inplace=True)
    df_.columns = df_.columns.get_level_values(1)

    n = df_.shape[0]
    limit = 20
    fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
    col1 = plotter.getColorList('viridis_r', n=n - 20)
    col2 = plotter.getColorList('tab20_r', n=20)
    # col3 = getColorList('tab20c',n=20)
    colors = col2 + col1
    colors = colors[::-1]
    N = len(colors)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined",
                                                               colors,
                                                               N=N)
    df_.T.plot.area(cmap=cmap, ax=ax)
    ax.set_xlim(0, None)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(ls=':', lw=0.7)
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.set_ylabel("call time (ms)")
    ax.set_title(
        "{}\n{} time per iteration for {}\nwithout cudaStreamSync and cudaMemcpyAsync"
        .format(target, title, logdir))
    # Reverse legend order
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1][:limit]
    labels = labels[::-1][:limit]
    leg = ax.legend(handles,
                    labels,
                    ncol=1,
                    loc='upper left',
                    frameon=False,
                    bbox_to_anchor=(1, 1.05))
    figfile = "{}_noOverlaps_nodatacopy.pdf".format(fig_)
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

# Area plot number of calls on CPU side
df_ = nooverlapdf[~nooverlapdf['corrID'].isin(memoryops['corrID'].unique()
                                              )].copy()
# Count number of calls per one iteration
df_ = df_.groupby(['param', 'name', 'iteration'], as_index=False).agg('size')

df_ = df_.groupby(['param',
                   'name']).agg('median').drop(['iteration'],
                                               axis=1).reset_index(drop=False)

df_ = df_.pivot(index="name", columns=['param'], values=['size']).fillna(0)
df_.loc[:, 'total'] = df_.sum(axis=1)

df_ = df_.sort_values(['total'])
df_.drop(['total'], axis=1, inplace=True)
df_.columns = df_.columns.get_level_values(1)

n = df_.shape[0]
limit = 20
fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
col1 = plotter.getColorList('viridis_r', n=n - 20)
col2 = plotter.getColorList('tab20_r', n=20)
# col3 = getColorList('tab20c',n=20)
colors = col2 + col1
colors = colors[::-1]
N = len(colors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined",
                                                           colors,
                                                           N=N)

df_.T.plot.area(cmap=cmap, ax=ax)
ax.set_xlim(0, None)
# ax.yaxis.set_major_locator(MultipleLocator(10))
# ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("number of calls)")
ax.set_title(
    "{}\nCPU calls number per iteration for {}\nwithout cudaStreamSync and cudaMemcpyAsync"
    .format(target, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles,
                labels,
                ncol=1,
                loc='upper left',
                frameon=False,
                bbox_to_anchor=(1, 1.05))
figfile = "CPUnumcalls_noOverlaps_nodatacopy.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# 2.7 Data transfer times

memcopy = longtimes[(longtimes['GPU side'] == True)
                    & (longtimes['corrID'] != 0)].copy()
memcopy = selectEventsByName(memcopy, ["cudaMemcpyAsync"])

df_ = memcopy.copy()
df_ = df_.groupby(['param', 'name',
                   'iteration']).agg('sum').reset_index(drop=False)
df_ = df_.groupby(['param', 'name']).agg('mean').reset_index(drop=False)
memcopy = df_.drop(['iteration'], axis=1).copy()
df_ = df_.pivot(index="name", columns=['param'], values=['duration']).fillna(0)

df_.columns = df_.columns.get_level_values(1)
df_ = df_ * 1000.

fig, ax = plt.subplots(figsize=(10, 5), dpi=144)

df_.T.plot(marker='o', ms=4, mfc='w', ax=ax)
ax.set_xlim(0, None)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("time (ms)")
ax.set_title("{}\n{} time per iteration for {}".format(target, event, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles,
                labels,
                ncol=1,
                loc='upper left',
                frameon=False,
                bbox_to_anchor=(1, 1.05))
figfile = "cudaMemcpyAsync.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# 3 Prepare CPUGPU DF

# 3.1 Calculate GPU time

callsandgapsGPU = None
df_ = longtimes[(longtimes['GPU side'] == True)].copy()
# Exclude memory operations
gpu_memoryops = selectEventsByName(df_, [
    "cudaStreamSynchronize", "cudaMemcpyAsync", "cudaEventQuery",
    "cudaStreamWaitEvent"
])
df_ = df_[~df_['corrID'].isin(gpu_memoryops['corrID'].unique())]
for mbs, mbsdf in df_.groupby('param', sort=True):
    for iteration, iterdf in mbsdf.groupby('iteration', sort=False):
        iterdf = iterdf.sort_values(['start']).reset_index(drop=True)
        iterdf.loc[:, 'prev'] = iterdf['end'].shift(1)
        iterdf.loc[:, 'gap'] = iterdf['start'] - iterdf['prev']
        if callsandgapsGPU is None:
            callsandgapsGPU = iterdf
        else:
            callsandgapsGPU = pd.concat([callsandgapsGPU, iterdf],
                                        ignore_index=True)

# Area plot GPU events

df_ = callsandgapsGPU.copy()
df_ = df_.groupby(['param', 'name',
                   'iteration']).agg('sum').reset_index(drop=False)
df_ = df_.groupby(['param', 'name']).agg('mean').reset_index(drop=False)
df_ = df_.pivot(index="name", columns=['param'], values=['duration']).fillna(0)
df_.loc[:, 'total'] = df_.sum(axis=1)
df_ = df_.sort_values(['total'])
df_.drop(['total'], axis=1, inplace=True)
df_.columns = df_.columns.get_level_values(1)
df_ = df_ * 1000.
n = df_.shape[0]
limit = 20
fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
col1 = plotter.getColorList('viridis_r', n=n - 20)
col2 = plotter.getColorList('tab20_r', n=20)
colors = col2 + col1
colors = colors[::-1]
N = len(colors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined",
                                                           colors,
                                                           N=N)

df_.T.plot.area(cmap=cmap, ax=ax)
ax.set_xlim(0, None)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title(
    "{}\nGPU events time per iteration for {}\nwithout memory operations".
    format(target, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles,
                labels,
                ncol=1,
                loc='upper left',
                frameon=False,
                bbox_to_anchor=(1, 1.05))
figfile = "GPUevents_Area.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

df_ = callsandgapsGPU.copy()
df_.loc[:, 'callandgap'] = df_['duration'] + df_['gap']
df_ = df_[['name', 'param', 'iteration', 'duration', 'callandgap']]
df_ = df_.groupby(['param', 'name',
                   'iteration']).agg('sum').reset_index(drop=False)
df_.drop(['iteration'], axis=1, inplace=True)
df_ = df_.groupby(['param', 'name']).agg('mean').reset_index(drop=False)
df_ = df_.groupby(['param']).agg('sum').reset_index(drop=False)
df_.set_index('param', drop=True, inplace=True)
callsandgapsGPUagg = df_.copy()
df_ = df_ * 1000.

fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
df_.plot(ax=ax)
ax.set_xlim(0, None)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel('time (ms)')
ax.set_title(
    "{}\nGPU events and gaps per iteration for {}\nwithout memory operations".
    format(target, logdir))
figfile = "GPUeventsAndGaps.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# 3.2  CPU time, memcpy time, GPU time

# print("Memcpy:\n", memcopy.head())
# print("CPU CUDA memcpy:\n", cpucudaMemcpy.head())
# print("No memory:\n", nomemorydf.head())
# print("Calls and Gaps:\n", callsandgapsGPUagg.head())

df_ = memcopy.copy()
df_ = df_[['param',
           'duration']].groupby(['param']).agg('sum').reset_index(drop=False)
CPUGPU = df_.rename({'param': 'batch', 'duration': 'memcpy'}, axis=1).copy()

df_ = nomemorydf.copy()
df_ = df_[['param', 'duration', 'callandgap'
           ]].groupby(['param']).agg('sum').reset_index(drop=False).rename(
               {
                   'param': 'batch',
                   'duration': 'CPU calls',
                   'callandgap': 'cpu_with_gaps'
               },
               axis=1)

CPUGPU = CPUGPU.merge(df_, on="batch")

df_ = callsandgapsGPUagg.reset_index(drop=False).copy()
df_ = df_.rename(
    {
        'param': 'batch',
        'duration': 'GPU',
        'callandgap': 'gpu_with_gaps'
    },
    axis=1)

CPUGPU = CPUGPU.merge(df_, on="batch")

df_ = cpucudaMemcpy[['param', 'duration'
                     ]].rename({
                         'param': 'batch',
                         'duration': 'cpusync'
                     },
                               axis=1)

CPUGPU = CPUGPU.merge(df_, on="batch").set_index('batch', drop=True)
CPUGPU.loc[:, 'CPU'] = CPUGPU['gpu_with_gaps'] - CPUGPU['cpusync']
print("CPUGPU:\n", CPUGPU.head())
df_ = CPUGPU * 1000.
fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
df_.plot(ax=ax)
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel('time (ms)')
ax.set_title(target)
figfile = "TraceEventsAndGaps.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Aggregate clean CPU time
df_ = clean_cpu.copy()
df_ = df_.groupby(['param']).agg('sum').drop(['start'], axis=1)
clean_cpu_agg = df_.copy()
clean_cpu_agg.head()

# Read IO time
dataio = pd.read_csv(os.path.join(dataiodir, "avg.csv"))
dataio.set_index("batch", drop=True, inplace=True)
dataio = dataio / 1000.  # Convert ms to s
dataio.loc[:, "median_read_time"] = dataio[
    "median_read_time"] * dataio.index.values
dataio.loc[:,
           "mean_read_time"] = dataio["mean_read_time"] * dataio.index.values

# Read time logs
clean_logs = pd.read_csv(time_logs)
if 'iterations' not in clean_logs.columns:
    print("Error: No iterations values in {}.".format(time_logs))
    print("Calculating itertime for dataset size of 50000")
    clean_logs['iterations'] = 50000 / clean_logs['batch']

clean_logs['itertime'] = clean_logs['time'] / clean_logs['iterations']
# Plottable value
clean_logs['value'] = clean_logs['itertime']
try:
    clean_logs = clean_logs.groupby(['batch']).agg('mean')
except:
    print("Error grouping clean logs by batch")
    print(time_logs)
    print(clean_logs.head())
    sys.exit(1)
clean_logs = clean_logs[['value']]
clean_logs.index.name = None

# Read DNNMark logs
dnnmark = pd.read_csv(dnnmark_logs)
dnnmark = dnnmark[['batch', 'CNN iteration time'
                   ]].rename({'CNN iteration time': 'DNNMark'}, axis=1)

lims = [300, 30]

for lim in lims:
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True, dpi=140)
    ax = axs[0]
    clean_logs.rename(columns={
        "value": "log time"
    }).loc[clean_logs.index <= lim, :].plot(marker='o',
                                            ms=4,
                                            mfc='white',
                                            ax=ax)
    dataio.loc[:lim, :].plot(y='mean_read_time',
                             use_index=True,
                             ms=4,
                             mfc='white',
                             marker='o',
                             ax=ax)
    dnnmark[dnnmark['batch'] <= lim].plot(x='batch',
                                          ms=4,
                                          mfc='white',
                                          marker='o',
                                          ax=ax)
    ax.set_ylim([0, None])
    # ax.set_xlim([0, None])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.grid(ls=':', lw=0.7)
    ax.set_title("{} iteration time".format(target))
    ax.set_ylabel('time (s)')

    ax = axs[1]
    clean_logs.rename(columns={
        "value": "log time"
    }).loc[clean_logs.index <= lim, :].plot(lw=1.5, ax=ax)
    dataio.loc[:lim, :].plot(y='mean_read_time',
                             use_index=True,
                             ls=":",
                             lw=1,
                             ax=ax)
    dnnmark[dnnmark['batch'] <= lim].plot(x='batch',
                                          ms=3,
                                          mfc='white',
                                          marker='o',
                                          lw=1,
                                          mew=0.5,
                                          ax=ax)
    CPUGPU.loc[:lim, ['CPU', 'GPU', 'memcpy']].plot(ms=3,
                                                    mfc='white',
                                                    marker='o',
                                                    lw=1,
                                                    mew=0.5,
                                                    ax=ax)
    CPUGPU.loc[:lim, ['cpu_with_gaps', 'gpu_with_gaps']].plot(ls=':',
                                                              ms=3,
                                                              mfc='white',
                                                              marker='o',
                                                              lw=1,
                                                              mew=0.5,
                                                              ax=ax)
    clean_cpu_agg.loc[:lim, :].plot(y="duration",
                                    ax=ax,
                                    label="only compute kernel calls")
    ax.set_title("{} iteration times".format(target))
    ax.set_ylim([0, None])
    ax.set_xlim([0, lim + 1])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.grid(ls=':', lw=0.7)
    ax.set_ylabel('time (s)')
    figfile = "itertimes_{}.pdf".format(lim)
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

print("Done.")
