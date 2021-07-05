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
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
import seaborn as sns
import numpy as np
from cycler import cycler
import itertools
from lib import plotter
import yaml
import argparse
from scipy import stats
from sklearn import linear_model
import time

version = "1.12e"
print("Analyser of series of nsys traces. v.{}.".format(version))
# Default parameters

target = None
targetdir = None
logdir = None
trace_name_pattern = 'nsys_trace_([0-9]+).json'
event_patterns = ['Iteration [123456789]+']
dataiodir = None
time_logs = None
dnnmark_logs = None

convert_traces = False

parser = argparse.ArgumentParser()
parser.add_argument("--conf", "-c", help="YAML file with parameters")
parser.add_argument("--reparse", action="store_true", default=False)
parser.add_argument("--nooverlaps", action="store_true", default=False)

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
        if "dataiodir" in conf:
            dataiodir = conf["dataiodir"]
        time_logs = conf["time_logs"]
        if 'dnnmark_logs' in conf:
            dnnmark_logs = conf["dnnmark_logs"]
    else:
        print("Cannot find config file {}.".format(args.conf))
        sys.exit(1)

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
def parseIteration(sorg):
    s = ''
    if type(sorg) == list:
        for a in sorg:
            if 'iteration' in a.lower():
                s = a
                break
    elif type(sorg) == str:
        try:
            s = sorg.strip(string.ascii_letters).strip(' ,')
        except Exception as e:
            print(e)
            print(s, type(s))
            return None
    elif np.isnan(sorg):
        return None
    else:
        return int(sorg)

    if 'iteration' not in s.lower():
        return None
    if len(s) < 1:
        return None
    try:
        i = int(s)
    except ValueError:
        s = [c for c in s if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        s = ''.join(map(str, s))
        if len(s) == 0:
            print("Cannot convert to int '{}' ({})".format(sorg, type(sorg)))
            print("No numbers found")
            return None
        try:
            i = int(s)
        except ValueError:
            print("Cannot convert to int '{}'".format(sorg))
            print("Value error in '{}'".format(s))
            return None
    return i


# Return DF with events (names) from the list of async event
def selectEventsByName(df, names):
    asyncEventsdf = None
    for aname in names:
        try:
            adf = df[(df['name'].notna()) & (df['name'].str.contains(aname))]
        except ValueError as e:
            print('Error in selectEventsByName')
            print(e)
            print("aname={}, df shape: {}".format(aname, df.shape))
            print("None name rows:")
            print(df[df['name'].isna()])
            raise (ValueError(e))
        if asyncEventsdf is None:
            asyncEventsdf = adf.copy()
        else:
            asyncEventsdf = pd.concat([asyncEventsdf, adf], ignore_index=False)
    return asyncEventsdf


def getEventsMask(df, names, column='name', debug=False):
    mask = None
    for name in names:
        mask1 = (df[column].str.contains(name))
        if mask is None:
            mask = mask1
        else:
            mask = (mask | mask1)
        if debug:
            print("Filtering  out {}/{} lines".format(len(mask[mask == True]), len(mask)))
            print("Remaining events: {}".format(df.loc[~mask, 'name'].unique()))
    return mask


# def getCleanCPUTime(df):
#     # Time without cuda... events
#     # Remove all cuda... and other not calculation events
#     cuda_names = [
#         'cudaMemcpy', 'cudaStream', 'cudaEvent', 'cudaMemset',
#         'cudaBindTexture', 'cudaUnbindTexture', 'cudaEventCreateWithFlags',
#         'cudaHostAlloc', 'cudaMalloc', 'CUPTI', 'cudaFree', 'cudaLaunchKernel'
#     ]
#     clean_cpu = df[(df['GPU side'] == False) & (df['corrID'] > 0)].copy()
#     cuda_events = selectEventsByName(clean_cpu, cuda_names)
#     clean_cpu = clean_cpu[~clean_cpu.index.isin(cuda_events.index)]
#     return clean_cpu


def getCleanCPUTime(df, debug=False):
    # Time without cuda... events
    # Remove all cuda... and other not calculation events
    cuda_names = [
        'cudaMemcpy', 'cudaStream', 'cudaEvent', 'cudaMemset', 'cudaBindTexture',
        'cudaUnbindTexture', 'cudaEventCreateWithFlags', 'cudaHostAlloc', 'cudaMalloc',
        'CUPTI', 'cudaFree', 'cudaLaunchKernel'
    ]
    clean_cpu = df[(df['GPU side'] == False) & (df['corrID'] > 0) &
                   (df['name'].notna())].copy()
    if debug:
        print("CPU events with corrID: {}".format(clean_cpu.shape[0]))
    mask = getEventsMask(clean_cpu, cuda_names, debug=debug)
    if debug:
        print("Final: Filtering  out {}/{} lines".format(len(mask[mask == True]),
                                                         len(mask)))
    clean_cpu = clean_cpu[~mask]
    if debug:
        print(clean_cpu['name'].unique())
    return clean_cpu


# Plot stacked areaplot of kernel times etc.
# Series name (distinguished by color) in column series.
# title - plot title
# limit - only largest series shown in the legend. The number of shown series is 'limit'.
# agg - aggregaion function
# yticks - tuple of major and minro Y-axis tick step
# values - column in df to use as the values for aggregation and plotting.
def plotArea(df, series='name', title=None, limit=20, agg='sum', yticks=None,
             values='duration', av_func='median', units=""):
    df_ = df.groupby([series, 'param', 'iteration'
                      ]).agg(agg).groupby([series,
                                           'param']).agg(av_func).reset_index(drop=False)
    df_ = df_.pivot_table(index=series, columns=['param'], values=[values]).fillna(0)
    df_.loc[:, 'total'] = df_.sum(axis=1)
    df_ = df_.sort_values(['total'])
    df_.drop(['total'], axis=1, inplace=True)
    df_.columns = df_.columns.get_level_values(1)

    n = df_.shape[0]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
    colors = plotter.getColorList('tab20', n=20)
    if n > 21:
        col1 = plotter.getColorList('viridis_r', n=n - 20)
        colors = colors + col1
    colors = colors[::-1]
    N = len(colors)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined", colors, N=N)

    df_.T.plot.area(cmap=cmap, ax=ax)
    ax.set_xlim(0, None)
    if yticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(yticks[0]))
        ax.yaxis.set_minor_locator(MaxNLocator(yticks[1]))
    else:
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(ls=':', lw=0.7)
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.set_ylabel("{} of {} ({})".format(agg, values, units))
    if title is not None:
        ax.set_title(title)
    # Reverse legend order
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1][:limit]
    labels = labels[::-1][:limit]
    leg = ax.legend(handles, labels, ncol=1, loc='upper left', frameon=False,
                    bbox_to_anchor=(1, 1.05))


# Create target dir
if not os.path.exists(targetdir):
    os.makedirs(targetdir)
print("Target dir", targetdir)

time0 = time.perf_counter()
# Read trace files
list_command = "ls -1 " + logdir
files = []
param_values = []
proc = subprocess.Popen(list_command.split(" "), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, encoding='utf8')
for line in iter(proc.stdout.readline, ''):
    line = line.strip(" \n")
    m = re.match(trace_name_pattern, line)
    if m:
        files.append(os.path.abspath(os.path.join(logdir, line)))
        param_values.append(m.group(1))

print('{} files in {}'.format(len(files), logdir))

results = None
for param, tracefile in zip(param_values, files):
    # events = ' '.join(event_patterns)
    # Read data from CSV file
    directory = os.path.dirname(tracefile)
    csvfile = ('').join(
        os.path.basename(tracefile).split('.')[:-1])  # Filename without extension
    csvfile = csvfile + '.csv'
    csvfile = os.path.join(directory, csvfile)
    if convert_traces or not os.path.exists(csvfile):
        print("Parcing", tracefile)
        # Run
        # python3 parseOneTrace.py -f $tracefile --events $events
        command = ['python3', 'parseOneTraceRapids.py', '-f', tracefile, '--events'
                   ] + event_patterns
        print(" ".join(command))
        p = subprocess.run(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                           bufsize=0, shell=False)
        if p.returncode == 0:
            print('Convertion finished OK')
        else:
            if p.stdout is not None:
                print(p.stdout.decode('utf-8'))
            print('ERROR')
            print(p.stderr.decode('utf-8'))

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

# Parse iteration numbers
slicesize = 1000000
maxdfsize = 2000000
time1 = time.perf_counter()
print("Reading done in {:.1f}s".format(time1 - time0))
print(results.head())

# Extract NVTX iteration times
# Filter
nvtx = results[(results['event_type'] == 1) & (results['end'].notna())].copy()
nvtx['NVTX'] = nvtx['NVTX'].str.replace(r'Iteration .*', 'Iteration')
nvtx['NVTX'] = nvtx['NVTX'] + "_NVTX"
nvtx = nvtx[['duration', 'NVTX',
             'param']].pivot_table(index='param', columns='NVTX', values='duration',
                                   aggfunc='median')

print("NVTX:")
print(nvtx.head())

# Remove Moving model time, because it is too large for the plot
nvtx.drop('Moving model to GPU_NVTX', axis=1, inplace=True)

print("mbs:", sorted(results['param'].unique()))
if results.shape[0] > maxdfsize:
    blocks = []
    for i in range(0, results.shape[0], slicesize):
        time2 = time.perf_counter()
        print(
            "\rParsing range {}-{}M lines of {:.1f}M ({:.1f})s.".format(
                i / 1000000., (i + slicesize) / 1000000., results.shape[0] / 1000000,
                time2 - time1), end='')
        block = results.iloc[i:i + slicesize].copy()
        iteration_mask = (block['NVTX'].notna()) & (block['NVTX'].str.contains(
            'iteration', flags=re.IGNORECASE))
        block.loc[iteration_mask, 'iteration'] = block["NVTX"].apply(parseIteration)
        # print("Block mbs:", sorted(block['param'].unique()))
        # print("Block NVTX:", block['NVTX'].unique())
        blocks.append(block)
    longtimes = blocks[0].append(blocks[1:])

else:
    longtimes = results.copy()
    iteration_mask = (longtimes['NVTX'].notna()) & (longtimes['NVTX'].str.contains(
        'iteration', flags=re.IGNORECASE))
    longtimes.loc[iteration_mask, 'iteration'] = longtimes["NVTX"].apply(parseIteration)

# longtimes['NVTX'] = longtimes['NVTX'].fillna('')
longtimes.drop(['NVTX'], axis=1, inplace=True)
longtimes[['param', 'corrID', 'Type']] = longtimes[['param', 'corrID',
                                                    'Type']].fillna(-1).astype(int)
longtimes[['iteration']] = longtimes[['iteration']].fillna(0).astype(int)
iterations = sorted(list(longtimes['iteration'].unique()))
longtimes = longtimes[longtimes['iteration'].isin(iterations[2:-2])]
print("Iterations: {}-{}".format(longtimes['iteration'].min(), longtimes.iteration.max()))
print("mbs:", sorted(longtimes['param'].unique()))

# Iteration time and its variablility
itertime = longtimes.groupby(['param', 'iteration']).agg({'start': 'min', 'end': 'max'})
itertime.loc[:, 'time'] = itertime.end - itertime.start
itertime = itertime.reset_index(drop=False).pivot(index='param', columns=['iteration'],
                                                  values='time')
itertime['median'] = itertime.median(axis=1)
itertime['mean'] = itertime.mean(axis=1)
itertime['min'] = itertime.min(axis=1)
itertime['max'] = itertime.max(axis=1)
iteration_variability = itertime[['mean', 'max', 'min']].copy()
iteration_variability.index.rename('batch', inplace=True)
iteration_variability.loc[:, 'delta'] = iteration_variability[
    'max'] - iteration_variability['min']
iteration_variability.loc[:, 'variability'] = iteration_variability[
    'delta'] / iteration_variability['mean'] * 100.
mean_var = iteration_variability['variability'].mean()
mean_delta = iteration_variability['delta'].mean()

# PLot iteration time and variabililty
fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=140)
ax = axs[0]
iteration_variability.plot(y=['mean', 'delta'], marker='o', ms=4, mfc='w', mew=0.5, ax=ax)
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel('Max-min (s)')
ax.set_title("Variability of Iteration Time from Traces", y=1.1, fontsize=14)
plt.suptitle('Mean range (max-min) {:.2f}s'.format(mean_delta), y=0.96)
ax = axs[1]
iteration_variability.plot(y='variability', marker='o', ms=4, mfc='w', mew=0.5, ax=ax)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel('variability (%)')
ax.set_title("Mean variability {:.2f}%".format(mean_var))
ax.text(-4, -4, logdir)

plt.tight_layout()
figfile = "iterationTime.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Time of CPU calls to compute kernels
# Remove all cuda... not calculation events
clean_cpu = getCleanCPUTime(longtimes)

# Remove outliers from CPU time
# Calculate Z-score per each kernel name
dfscored = None
for name, dfname in clean_cpu.groupby(['name']):
    dfname.loc[:, 'zscore'] = stats.zscore(dfname['duration'])
    dfname.loc[:, 'median'] = dfname['duration'].median()
    if dfscored is None:
        dfscored = dfname
    else:
        dfscored = pd.concat([dfscored, dfname], ignore_index=True)

# Replace outliers duration with median time for the kernel name
dfscored.loc[dfscored['zscore'] > 3, 'duration'] = dfscored['median']

memusage = int(clean_cpu.memory_usage(deep=True).sum())
print("clean_cpu DF memory usage: {} MB.".format(memusage / 100000.))

if memusage > 100000000:
    print("clean_cpu size too large for the scatter plot of event durations.")
else:
    # Scatterplot of CPU event durations for clean_cpu and dfscored
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(clean_cpu.param, clean_cpu.duration, marker='o', ms=3, mew=0, alpha=0.5,
                linestyle='', label=name)
    axs[1].plot(dfscored.param, dfscored.duration, marker='o', ms=3, mew=0, alpha=0.5,
                linestyle='', label=name)
    axs[0].set_ylabel("time (s)")
    axs[0].set_title("Compute kernel call durations\nAll")
    axs[1].set_title("Compute kernel call durations\nZscore > 3")

    axs[0].grid(ls=':')
    axs[1].grid(ls=':')
    figfile = "CPUcomputeKernelDurations.pdf"
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

clean_cpu = dfscored

# Plot Area plots of CPU events (for compute kernels only)

df_ = clean_cpu[['name', 'duration', 'param', 'iteration']].copy()
df_['duration'] = df_['duration'] * 1000.
plotArea(df_, series='name',
         title="Compute kernel calls, median time per iteration\n{}".format(logdir),
         yticks=(4, 20), units="ms")
figfile = "CPUcomputeKernelCallsTimeArea.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

df_ = clean_cpu[['name', 'duration', 'param', 'iteration']].copy()
plotArea(df_, series='name',
         title="Compute kernel calls number per iteration\n{}".format(logdir),
         agg='count', values='name')
figfile = "CPUcomputeKernelCallsCountArea.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# CPU-GPU syncro time
event = "cudaMemcpyAsync"
syncevents = selectEventsByName(longtimes[longtimes['GPU side'] == False], [event])

df_ = syncevents.copy()
ignorecolumns = [c for c in ['start', 'end', 'corrID', 'prev', 'gap'] if c in df_.columns]
df_ = df_.drop(ignorecolumns, axis=1)
df_ = df_.groupby(['param', 'name', 'iteration']).agg('sum').reset_index(drop=False)
print("MBS in syncevents:")
print(' '.join([str(m) for m in df_['param'].unique()]))
if 25 not in df_['param'].unique():
    import pdb
    pdb.set_trace()
df_ = df_.groupby(['param', 'name']).agg('median').reset_index(drop=False)
cpucudaMemcpy = df_.copy()
df_['duration'] = df_['duration'] * 1000.
df_ = df_.pivot(index="name", columns=['param'], values=['duration']).fillna(0)
df_.columns = df_.columns.get_level_values(1)

fig, ax = plt.subplots(figsize=(10, 5), dpi=144)

df_.T.plot(marker='o', ms=4, mfc='w', ax=ax)
ax.set_xlim(0, None)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title("{} time per iteration for\n{}".format(event, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
# handles = handles[::-1][:limit]
# labels = labels[::-1][:limit]
leg = ax.legend(handles, labels, ncol=1, loc='upper left', frameon=False,
                bbox_to_anchor=(1, 1.05))
figfile = "CPUGPUsync_cudaMemcpyAsync.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

# Data Copy Time (Memcpy)
print("Events without names in longtimes")
print(longtimes[longtimes['name'].isna()].shape[0])
memcopy = longtimes[(longtimes['GPU side'] == True) & (longtimes['corrID'] != 0)].copy()
memcopy = selectEventsByName(memcopy, ["cudaMemcpyAsync"])
print("Memcpy:\n", memcopy.head())
memcopyagg = memcopy[['param', 'iteration', 'duration']].groupby(
    ['param', 'iteration'], as_index=False).agg('sum').groupby(
        ['param'], as_index=False).agg('median').drop('iteration', axis=1)
memcopyagg = memcopyagg.rename({'duration': 'memcopy', 'param': 'batch'}, axis=1)
print("Memcpyagg:\n", memcopyagg.head())

# 3 Prepare CPUGPU DF

# 3.1 Calculate GPU time
print("Calculating GPU time...")
callsandgapsGPU = None
df_ = longtimes[(longtimes['GPU side'] == True)].copy()
# Exclude memory operations
gpu_memoryops = selectEventsByName(
    df_,
    ["cudaStreamSynchronize", "cudaMemcpyAsync", "cudaEventQuery", "cudaStreamWaitEvent"])
df_ = df_[~df_['corrID'].isin(gpu_memoryops['corrID'].unique())]
for mbs, mbsdf in df_.groupby('param', sort=True):
    for iteration, iterdf in mbsdf.groupby('iteration', sort=False):
        iterdf = iterdf.sort_values(['start']).reset_index(drop=True)
        iterdf.loc[:, 'prev'] = iterdf['end'].shift(1)
        iterdf.loc[:, 'gap'] = iterdf['start'] - iterdf['prev']
        if callsandgapsGPU is None:
            callsandgapsGPU = iterdf
        else:
            callsandgapsGPU = pd.concat([callsandgapsGPU, iterdf], ignore_index=True)

# Area plot GPU events
print("Making area plots...")
df_ = callsandgapsGPU.copy()
df_ = df_.groupby(['param', 'name', 'iteration']).agg('sum').reset_index(drop=False)
df_ = df_.groupby(['param', 'name']).agg('mean').reset_index(drop=False)
df_ = df_.pivot(index="name", columns=['param'], values=['duration']).fillna(0)
df_.loc[:, 'total'] = df_.sum(axis=1)
df_ = df_.sort_values(['total'])
df_.drop(['total'], axis=1, inplace=True)
df_.columns = df_.columns.get_level_values(1)
df_ = df_ * 1000.
n = df_.shape[0]
limit = 20

col1 = plotter.getColorList('viridis_r', n=n - 20)
col2 = plotter.getColorList('tab20_r', n=20)
colors = col2 + col1
colors = colors[::-1]
N = len(colors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("combined", colors, N=N)

fig, axs = plt.subplots(3, 1, figsize=(12, 15), dpi=144)
ax = axs[0]
df_.T.plot.area(cmap=cmap, ax=ax)
ax.set_xlim(0, None)
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title("{}\nGPU events time per iteration for {}\nwithout memory operations".format(
    target, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles, labels, ncol=1, loc='upper left', frameon=False,
                bbox_to_anchor=(1, 1.05))

# BN layers only
ax = axs[1]
df_.filter(like="bn_", axis=0).T.plot.area(cmap=cmap, ax=ax)
ax.set_xlim(0, None)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title("{}\nBN layers kernels time for {}\nwithout memory operations".format(
    target, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles, labels, ncol=1, loc='upper left', frameon=False,
                bbox_to_anchor=(1, 1.05))

# Pooling layers only
ax = axs[2]
df_.filter(like="pool_", axis=0).T.plot.area(cmap=cmap, ax=ax)
ax.set_xlim(0, None)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(ls=':', lw=0.7)
ax.grid(ls=':', lw=0.3, which='minor')
ax.set_ylabel("call time (ms)")
ax.set_title("{}\nPooling layers kernels time for {}\nwithout memory operations".format(
    target, logdir))
# Reverse legend order
handles, labels = ax.get_legend_handles_labels()
handles = handles[::-1][:limit]
labels = labels[::-1][:limit]
leg = ax.legend(handles, labels, ncol=1, loc='upper left', frameon=False,
                bbox_to_anchor=(1, 1.05))

plt.tight_layout()
figfile = "GPUevents_Area.pdf"
figfile = os.path.join(targetdir, figfile)
plt.savefig(figfile, bbox_inches="tight")
print("Saved", figfile)

df_ = callsandgapsGPU.copy()
df_.loc[:, 'callandgap'] = df_['duration'] + df_['gap']
df_ = df_[['name', 'param', 'iteration', 'duration', 'callandgap']]
df_ = df_.groupby(['param', 'iteration']).agg('sum').reset_index(drop=False)
df_ = df_.groupby(['param']).agg('mean').reset_index(drop=False)
df_.drop(['iteration'], axis=1, inplace=True)
df_.rename({'param': 'batch'}, axis=1, inplace=True)
callsandgapsGPUagg = df_.copy()
print("GPU calls and gaps")
print(callsandgapsGPUagg.head(2))

# 3.2  CPU time, memcpy time, GPU time

CPUGPU = memcopyagg.copy()

df_ = callsandgapsGPUagg.copy()
df_ = df_.rename({'duration': 'GPU', 'callandgap': 'GPU with gaps'}, axis=1)

CPUGPU = CPUGPU.merge(df_, on="batch")

df_ = cpucudaMemcpy[['param',
                     'duration']].rename({
                         'param': 'batch',
                         'duration': 'cpusync'
                     }, axis=1)

CPUGPU = CPUGPU.merge(df_, on="batch")

print("CPUGPU:\n", CPUGPU.head())

# # Aggregate clean CPU time

df_ = clean_cpu[['name', 'duration', 'param', 'iteration']].copy()
df_ = df_.groupby(['param', 'iteration'], as_index=False).agg('sum').groupby(
    ['param'], as_index=False).agg('min').drop(['iteration'], axis=1)
clean_cpu_agg = df_.copy()
clean_cpu_agg = clean_cpu_agg.rename({'param': 'batch', 'duration': 'clean CPU'}, axis=1)

print('Clean CPU time df\n', clean_cpu_agg.head())
CPUGPU = CPUGPU.merge(clean_cpu_agg, on="batch").set_index('batch', drop=True)

# Read IO time. CSV file avg.csv has times to read one image by one worker.
dataio = None
if dataiodir is not None and len(dataiodir) > 0:
    dataio = pd.read_csv(os.path.join(dataiodir, "avg.csv"))
    dataio[['median_read_time',
            'mean_read_time']] = dataio[['median_read_time',
                                         'mean_read_time']].astype(float)
    dataio['batch'] = dataio['batch'].astype(int)
    dataio.loc[:, "median_read_time"] = dataio["median_read_time"] * dataio['batch']
    dataio.loc[:, "mean_read_time"] = dataio["mean_read_time"] * dataio['batch']
    dataio.set_index("batch", drop=True, inplace=True)
    dataio = dataio / 1000.  # Convert ms to s
    print("Images read time (IO time)")
    print(dataio.head())

# Read time logs
clean_logs = None
if time_logs is not None:
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
dnnmark = None
if dnnmark_logs is not None and len(dnnmark_logs) > 0:
    dnnmark = pd.read_csv(dnnmark_logs)
    dnnmark = dnnmark[['batch',
                       'CNN iteration time']].rename({'CNN iteration time': 'DNNMark'},
                                                     axis=1)

# Aggregate nonverlapdf
# Sum time for each iteration
if args.nooverlaps:
    nooverlap_agg = nooverlapdf[['param', 'iteration', 'duration', 'gap']].groupby(
        ['param', 'iteration'], as_index=False).sum().rename({'param': 'batch'}, axis=1)
    nooverlap_agg = nooverlap_agg.groupby(['batch'], as_index=True).agg('min')
    nooverlap_agg.loc[:,
                      'timewithgaps'] = nooverlap_agg['duration'] + nooverlap_agg['gap']
    nooverlap_agg.drop(['iteration', 'gap'], axis=1, inplace=True)

    print("Aggregated Nooverlapdf\n", nooverlap_agg.head())

# Merge tables and save CSV
if clean_logs is not None:
    df = pd.concat([CPUGPU, clean_logs.rename({'value': 'log_time'}, axis=1)], axis=1)
else:
    df = CPUGPU.rename({'value': 'log_time'}, axis=1).copy()
if args.nooverlaps:
    df = pd.concat([
        df,
        nooverlap_agg.rename(
            {
                'duration': 'nooverlap CPU min',
                'timewithgaps': 'nooverlap CPU with gaps'
            }, axis=1)
    ], axis=1)
# Sort columns
df = df.reindex(sorted(df.columns), axis=1)

print("MERGED DF")
# print(" ".join(df.columns))
print(df.head())
# df = df.rename({'index': 'batch'}, axis=1)
# Save merged DF
csvfile = "datamerged.csv"
path = os.path.join(targetdir, csvfile)
df.to_csv(path, index=False)
print("Saved data to {}.".format(path))

withgapscolumns = [c for c in df.columns if 'gaps' in c]
nogapscolumns = [c for c in df.columns if 'gaps' not in c]

colors = plotter.getColorList('tab10', n=10) + plotter.getColorList('Set2', n=8)
plt.rc('axes', prop_cycle=(cycler('color', colors)))

lims = [df.index.max(), 30]

for lim in lims:
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True, dpi=140)
    ax = axs[0]
    if 'log_time' in df.columns:
        df.loc[:lim, ['log_time']].plot(ms=4, mfc='white', marker='o', ax=ax)
    if dataio is not None:
        dataio.loc[:lim, :].plot(y='mean_read_time', use_index=True, ms=4, mfc='white',
                                 marker='o', ax=ax)
    if dnnmark is not None:
        dnnmark[dnnmark['batch'] <= lim].plot(x='batch', ms=4, mfc='white', marker='o',
                                              ax=ax)

    ax.set_ylim([0, None])
    # ax.set_xlim([0, None])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(MultipleLocator(0.05))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.grid(ls=':', lw=0.7)
    ax.set_title("{} iteration time".format(target))
    ax.set_ylabel('time (s)')
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1, 1.05))

    ax = axs[1]
    df.loc[:lim, nogapscolumns].plot(ms=4, lw=1, mfc='white', marker='o', ax=ax)
    df.loc[:lim, withgapscolumns].plot(ms=2.5, lw=0.6, mfc='white', mew=0.5, marker='o',
                                       ax=ax)
    nvtx.loc[:lim, :].plot(lw=1, marker='o', ms=4, mfc='white', mew=0.8, ls='--', ax=ax)
    # df.loc[:lim, [
    #     'log_time', 'clean CPU', "clean CPU with gaps", 'nooverlap CPU',
    #     'nooverlap CPU with gaps', 'CPU (gpu-sync)', 'GPU', 'GPU with gaps',
    #     'memcpy', 'cpusync'
    # ]].plot(ms=4, mfc='white', marker='o', ax=ax)
    if dataio is not None:
        dataio.loc[:lim, :].plot(y='mean_read_time', use_index=True, ls=":", lw=1, ax=ax)
    if dnnmark is not None:
        dnnmark[dnnmark['batch'] <= lim].plot(x='batch', ms=3, mfc='white', marker='o',
                                              ls=":", lw=1, mew=0.5, ax=ax)

    ax.set_title("{} iteration times".format(target))
    ax.set_ylim([0, None])
    ax.set_xlim([0, lim + 1])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.grid(ls=':', lw=0.7)
    ax.set_ylabel('time (s)')
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1, 1.05))
    figfile = "itertimes_{}.pdf".format(lim)
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

# log time as a function of CPU time + memcpy
if 'log_time' in df.columns:
    cputime = cpucudaMemcpy[['param', 'duration']].copy()
    # Set ceiling for the sync time (max possible = 1ms)
    maxsynctime = min(cputime.iloc[0]['duration'] * 1.05, 0.001)
    cputime = cputime[cputime['duration'] < maxsynctime]
    print("Sync time that is considered to indicate non-GPU time")
    print(cputime)
    maxmbs = cputime.param.max()
    print("Max MBS for LR approximation={}".format(maxmbs))
    CPUcolumns = [c for c in df.columns if 'CPU' in c]
    n = len(CPUcolumns) + 1
    fig, axs = plt.subplots(n, 1, figsize=(10, 3 * n), dpi=140)
    for i, col in enumerate(CPUcolumns):
        sum_column = col + " + memcopy"
        df.loc[:, sum_column] = df[col] + df['memcopy']
        ax = axs[i]
        df.loc[:maxmbs].plot(x=sum_column, y='log_time', marker='o', ms=5, mfc='w', ax=ax)
        for i, r in df.loc[:maxmbs].iterrows():
            ax.annotate(i, (r[sum_column], r['log_time']), xytext=(-5, 5),
                        textcoords='offset points', fontsize=7)
        ax.set_ylabel('log time (s)')
        ax.set_xlabel('{} (s)'.format(sum_column))
        # ax.xaxis.set_major_locator(MultipleLocator(0.005))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_major_locator(MultipleLocator(0.01))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.001))
        ax.grid(ls=':', lw=0.3, which='minor')
        ax.grid(ls=':', lw=0.7)
        ax.set_title("logtime({}) on MBS <={}\n{}".format(sum_column, maxmbs, target))

    ax = axs[n - 1]
    df.loc[:maxmbs].plot(x='memcopy', y='log_time', marker='o', ms=5, mfc='w', ax=ax)
    for i, r in df.loc[:maxmbs].iterrows():
        ax.annotate(i, (r['memcopy'], r['log_time']), xytext=(-5, 5),
                    textcoords='offset points', fontsize=7)
    ax.set_ylabel('log time (s)')
    ax.set_xlabel('memcopy (s)')
    ax.set_title("logtime(memcopy)\n{}".format(target))
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.grid(ls=':', lw=0.7)

    plt.tight_layout()
    figfile = "itertime_asfunction_ofCPUtime.pdf"
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

    # Approximate itertime(memcopy) with LR
    X = df.loc[:maxmbs]['memcopy'].values
    X = np.reshape(X, (-1, 1))
    Y = df.loc[:maxmbs]['log_time'].values
    lrmodel = linear_model.LinearRegression().fit(X, Y)

    f = lrmodel.predict(X)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(X, Y, label='Itertime(memcopy)', marker='o', ms=4, mfc='white', mew=0.7)
    ax.plot(X, f, label="LR", ls=':')
    ax.set_xlabel('memcopy (s)')
    ax.set_ylabel('iteration time (s)')
    ax.legend()
    fig.suptitle(
        "Itertime(memcopy)\nLR = $a{:.2f} {:+.2f}$".format(lrmodel.coef_[0],
                                                           lrmodel.intercept_),
        fontsize=14, y=1.02)
    ax.set_title(logdir, fontsize=10)
    ax.grid(ls=':', lw=0.5)
    figfile = "Approximate_itertime_func_memcopy.pdf"
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

    meanCPUtime = df.loc[:maxmbs]['clean CPU'].mean()
    print("Mean CPU time = {:.5f}s".format(meanCPUtime))

    X = (df.loc[:maxmbs]['memcopy'] + meanCPUtime).values
    X = np.reshape(X, (-1, 1))
    Y = df.loc[:maxmbs]['log_time'].values
    lrmodel = linear_model.LinearRegression().fit(X, Y)

    f = lrmodel.predict(X)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(X, Y, label='Itertime(CPU+memcopy)', marker='o', ms=4, mfc='white', mew=0.7)
    ax.plot(X, f, label="LR", ls=':')
    ax.set_xlabel('CPU + memcopy (s)')
    ax.set_ylabel('iteration time (s)')
    ax.legend()
    fig.suptitle(
        "Itertime(CPU+memcopy)\nLR = $a{:.2f} {:+.2f}$".format(lrmodel.coef_[0],
                                                               lrmodel.intercept_),
        fontsize=14, y=1.02)
    ax.set_title(logdir, fontsize=10)
    ax.grid(ls=':', lw=0.5)
    ax.text(0.01, 0.8, "Mean CPU time = {:.5f}s".format(meanCPUtime),
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    figfile = "Approximate_itertime_func_CPUmean_plus_memcopy.pdf"
    figfile = os.path.join(targetdir, figfile)
    plt.savefig(figfile, bbox_inches="tight")
    print("Saved", figfile)

time2 = time.perf_counter()
print("All done in {:.1f}s.".format(time2 - time0))
