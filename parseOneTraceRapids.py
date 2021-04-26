#!/usr/bin/env python3

# Extract data from one JSON trace file.
# 1. Find all events in 'names' and 'nvtx' DF-s matching patterns from the --events option.
# 2. For API events names found in 'names', search all API events by 'id' field of 'names'
#      matching 'TraceProcessEvent.name' field of 'traces' DF.
# 3. Find NVTX ranges encompassing each API event.
# 4. Find CUDA kernels (GPU side) for each API event. Lookup CUDA API calls (CPU side) within
#    the time range of each API event, than lookup CUDA kernels (GPU side) by correlation IDs.
# 5. Find all NVTX ranges matching patterns from --events.
# 6. For each NVTX range find CUDA API calls and CUDA kernels.
# 7. Store two rows: for the NVTX event on CPU side, and for corresponding CUDA kernels on GPU side.

# Columns in the returned (stored in a CSV file) dataframe:
# 'name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side'
# name - name of API call, CUDA kernel or NVTX range matching patterns from --event option,
# start,end - start and end time in seconds,
# duration in seconds,
# NVTX - nvtx ranges encompassing event,
# corrID - correlation ID (links CUDA API calls and CUDA kernels),
# GPU side - true if the event runs on GPU side.

# (C) 2021 Peter Bryzgalov @ Stair Laboratory, Chiba Institute of Technology, Japan

import os
import argparse
import pandas as pd
import json
from pandas import json_normalize
import numpy as np
import re
import sys
from multigpuexec import message
import datetime
import time
import pprint
import cudf
import pyarrow as pa
from lib import lib3

pp = pprint.PrettyPrinter(indent=4)

ver = '0.20a-gpu'

print('NVIDIA NSight Systems JSON trace parser v{}.'.format(ver))

parser = argparse.ArgumentParser(
    description=
    'Extracts time of CPU-side and GPU-side events and NVTX regions.')
parser.add_argument("--file",
                    '-f',
                    default=None,
                    required=True,
                    help="Trace filename to parse.")
parser.add_argument(
    "--events",
    default=None,
    required=False,
    nargs='*',
    help="Event name patterns. Multiple space-separated values possible.")
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()
debug = args.debug
event_name_patterns = args.events
extradebug = False
if debug:
    print(datetime.datetime.now())

# Start timer
start_t = time.perf_counter()
time1 = None
time2 = None
# Read Trace file
tmpfile = '.'.join(args.file.split('.')[:-1]) + "_raw.csv"
if os.path.exists(tmpfile):
    print("Reading from raw CSV file {}".format(tmpfile))
    df = pd.read_csv(tmpfile)
else:
    print("Reading {}...".format(args.file))
    if not os.path.exists(args.file):
        print("File {} not found.".format(args.file))
    data = [json.loads(line) for line in open(args.file, 'r')]
    time1 = time.perf_counter()
    print("File {} loaded in {:.2f}s".format(args.file, time1 - start_t))
    df = json_normalize(data)
    time2 = time.perf_counter()
    print("JSON normalized in {:.2f}s".format(time2 - time1))

    df.to_csv(tmpfile, index=False)
    print("Saved DF to raw CSV file {}".format(tmpfile))

time1 = time.perf_counter()
print("IO done in {:.2f}s".format(time1 - start_t))
int_columns = [
    'NvtxEvent.Timestamp', 'NvtxEvent.EndTimestamp', 'CudaEvent.startNs',
    'CudaEvent.endNs', 'CudaEvent.correlationId', 'CudaEvent.sync.eventId',
    'TraceProcessEvent.correlationId', 'TraceProcessEvent.name',
    'TraceProcessEvent.startNs', 'TraceProcessEvent.endNs', 'id'
]
int_columns = [c for c in int_columns if c in df.columns]
for c in int_columns:
    df[c] = df[c].fillna(-1).astype(int).replace(-1, np.nan)

df.dropna(axis=1, how='all')
# print("Read {} rows from {}.".format(df.shape[0], args.file))

# Mark events
OneDF = df.copy()
# Use column "event_type" for distinguish event types within OneDF
OneDF.loc[:, 'event_type'] = 0
# Event type code:
# 1: NVTX
# 2: Trace
# 4: Kernel

# Combined types ! Пересечений не обнаружено
# Trace + Kernel: 6
# NVTX + Trace + Kernel: 7


def markNVTX(df):
    df.loc[df['NvtxEvent.Timestamp'].notna(),
           'event_type'] = df['event_type'] + 1
    # Set standard start, end and duration columns in seconds
    df.loc[df['event_type'] == 1, 'start'] = df['NvtxEvent.Timestamp'] * 1e-9
    df.loc[df['event_type'] == 1, 'end'] = df['NvtxEvent.EndTimestamp'] * 1e-9
    return df


def markTraces(df):
    indexes = df['TraceProcessEvent.startNs'].notna()
    df.loc[indexes, 'event_type'] = df['event_type'] + 2
    df.loc[indexes, 'name_id'] = df['TraceProcessEvent.name']
    # Unified column for correlation ID
    df.loc[indexes, 'correlationID'] = df['TraceProcessEvent.correlationId']
    # Set standard start, end and duration columns in seconds
    df.loc[indexes, 'start'] = df['TraceProcessEvent.startNs'] * 1e-9
    df.loc[indexes, 'end'] = df['TraceProcessEvent.endNs'] * 1e-9
    return df


def markKernels(df):
    indexes = df['CudaEvent.startNs'].notna()
    df.loc[indexes, 'event_type'] = df['event_type'] + 4
    df.loc[indexes, 'correlationID'] = df['CudaEvent.correlationId']
    df.loc[(df['CudaEvent.startNs'].notna()) & (df['name_id'].isna()),
           'name_id'] = df['CudaEvent.kernel.shortName']
    # Set standard start and end columns in seconds
    df.loc[indexes, 'start'] = df['CudaEvent.startNs'] * 1e-9
    df.loc[indexes, 'end'] = df['CudaEvent.endNs'] * 1e-9
    return df


OneDF = markNVTX(OneDF)
OneDF = markTraces(OneDF)
OneDF = markKernels(OneDF)

OneDF.loc[:, 'duration'] = OneDF['end'] - OneDF['start']

# Drop unused DF columns
nvtx_columns = ['NvtxEvent.Text']
trace_columns = [
    'Type', 'TraceProcessEvent.eventClass', 'TraceProcessEvent.name'
]
kernel_columns = [
    'CudaEvent.memcpy.sizebytes', 'CudaEvent.memcpy.copyKind',
    'CudaEvent.kernel.shortName'
]
other_columns = [
    'event_type', 'correlationID', 'name_id', 'start', 'end', 'duration',
    'value'
]
columns = nvtx_columns + trace_columns + kernel_columns + other_columns
OneDF = OneDF[[c for c in columns if c in OneDF.columns]]
OneDF.dropna(how='all', axis=0, inplace=True)
OneDF = OneDF[OneDF['event_type'] != 0]

# nvtx DF
# Select only NVTX regions (not marks) in 'nvtx' DF
nvtx = OneDF[OneDF.event_type == 1].dropna(how='all', axis=1)
nvtx = nvtx[nvtx['end'].notna()]

# Names DF
names = df[df['value'].notna()].dropna(axis=1, how='all')


# Cross Join function
def crossJoinAndMerge(eventsgdf, nvtxgdf, debug=False):
    snvtx = nvtxgdf.reset_index(drop=False).rename({'index': 'nvtxid'}, axis=1)
    snvtx = snvtx.drop(['NvtxEvent.Text'], axis=1)
    # cross join
    eventsgdf['key'] = 1
    snvtx['key'] = 1
    merged = eventsgdf.merge(snvtx,
                             how="outer",
                             on="key",
                             suffixes=['', '_nvtx'])
    del merged["key"]
    # filter
    mask = (merged.start_nvtx <= merged.start) & (merged.end_nvtx >=
                                                  merged.end)
    del merged["start_nvtx"], merged["end_nvtx"]

    merged = merged[mask]
    if debug:
        print("Merged after filtering shape:", merged.shape)
        nvtxmin = snvtx.start.min()
        nvtxmax = snvtx.end.max()
        eventmin = eventsgdf.start.min()
        eventmax = eventsgdf.end.max()
        print("Events range: {:.2f}-{:.2f}".format(eventmin, eventmax))
        print("NVTX range  : {:.2f}-{:.2f}".format(nvtxmin, nvtxmax))
    if merged.shape[0] == 0:
        return None
    # Merge NVTX names
    merged = merged.merge(nvtxgdf[['NvtxEvent.Text']],
                          left_on='nvtxid',
                          right_index=True,
                          how='left')
    merged.drop(['nvtxid'], axis=1, inplace=True)
    # Aggregate
    try:
        merged = merged.groupby(['name_id', 'start', 'end', 'event_type'],
                                as_index=False).agg({'NvtxEvent.Text': list})
    except Exception as e:
        print(e)
        print("Merged:")
        pp.pprint(merged.head())
        return None
    return merged


# Prepare for Cross Join
onegdf = cudf.DataFrame.from_pandas(
    OneDF[(OneDF.event_type == 2) | (OneDF.event_type == 4)].drop_duplicates())
onegdf = onegdf.drop(['NvtxEvent.Text'], axis=1).sort_values(['start'])
onegdf_ = onegdf[['name_id', 'event_type', 'start', 'end']]
nvtxgdf = cudf.DataFrame.from_pandas(nvtx[['start', 'end', 'NvtxEvent.Text']])
namesgdf = cudf.DataFrame.from_pandas(names[['id', 'value'
                                             ]].rename({'value': 'name'},
                                                       axis=1))

# Cross Join to find enclosing NVTX ranges
# Merge back into onegdf -> result
slicesize = 50000
length = onegdf_.shape[0]
dfs = []
# mergedonegdf = onegdf.copy()
for i in range(0, length, slicesize):
    eventblock = onegdf_.iloc[i:i + slicesize]
    df = crossJoinAndMerge(eventblock, nvtxgdf)
    if df is None:
        block = onegdf_.iloc[i:i + slicesize]
        if 'NvtxEvent.Text' not in block.columns:
            block['NvtxEvent.Text'] = cudf.Series(
                pa.array([], type=pa.list_(pa.string())))
        df = block

    # Merge with onegdf
    df = df.sort_values(['start'])
    try:
        mergedgdf = onegdf.iloc[i:i + slicesize].merge(
            df, on=['event_type', 'start', 'end', 'name_id'], how='left')
    except ValueError:
        print("ValueError")
        pp.pprint(mergedgdf.iloc[i:i + slicesize].head())
        print(mergedgdf.shape)
        pp.pprint(df.head())
        print(df.shape)
        break

    dfs.append(mergedgdf)

eventsgdf = dfs[0].append(dfs[1:], ignore_index=True)

time2 = time.perf_counter()
print("Merging events and NVTX ranges done in {:.2f}s".format(time2 - time1))

# Fill missing GPU-side event names with CPU-side event names
# Select corrID of events with missing names
corrids = eventsgdf[eventsgdf['name_id'].isna()]['correlationID'].unique()
gdf_ = eventsgdf.copy().sort_values(['correlationID', 'event_type'])
mask = gdf_['correlationID'].isin(corrids)
# GDF with good names
gdf_good = gdf_[~mask]
# GDF with missing names
gdf_noname = gdf_[mask]

print("Fill missing GPU side event names (missing/good) {}/{}.".format(
    gdf_noname.shape[0], gdf_good.shape[0]))

# Fill missing names using temporary column namecopy.
gdf_ = gdf_noname[['event_type', 'correlationID', 'name_id']]
gdf_['namecopy'] = gdf_.shift(1)['name_id']
gdf_['nameid2'] = gdf_['name_id']
mask = gdf_['nameid2'].isna()
gdf_.loc[mask, 'nameid2'] = gdf_['namecopy']
gdf_.drop(['name_id', 'namecopy'], axis=1, inplace=True)
gdf_.rename({'nameid2': 'name_id'}, axis=1, inplace=True)

# Append GDFs with missing and good names
eventsgdf = gdf_good.append([gdf_noname])

time1 = time.perf_counter()
print("Filled missing names in {:.2f}s".format(time1 - time2))

# Merge namesdf: replace name_id numbers with names
eventsgdf = eventsgdf.merge(namesgdf,
                            left_on='name_id',
                            right_on='id',
                            how='left').drop(['name_id', 'id'], axis=1)

events = eventsgdf.to_pandas()
# Add NVTX ranges to result DF
events = events.append(OneDF[OneDF.event_type == 1],
                       ignore_index=True).sort_values(['start']).rename(
                           {'NvtxEvent.Text': 'NVTX'}, axis=1)

final_columns = [
    'Type', 'name', 'start', 'end', 'duration', 'NVTX', 'correlationID',
    'event_type'
]
print(events.columns)
events = events[final_columns]

# Compatibility measures
# GPU side column
events['GPU side'] = None
events.loc[events['event_type'] == 2, 'GPU side'] = False
events.loc[events['event_type'] == 4, 'GPU side'] = True
# corrID
events.rename({'correlationID': 'corrID'}, axis=1, inplace=True)

if events.shape[0] > 0:
    events.sort_values(by=['start', 'corrID'], inplace=True, ignore_index=True)
    print()
    print('Event DF head')
    pp.pprint(events.sort_values(['corrID'], ignore_index=True).head())
    print('{}  rows'.format(events.shape[0]))
    print('-' * 50)
    directory = os.path.dirname(args.file)
    filename = ('.').join(os.path.basename(
        args.file).split('.')[:-1])  # Filename without extension
    filename = filename + '.csv'
    filename = os.path.join(directory, filename)
    events.to_csv(filename, index=False)
    print('Saved to {}.'.format(filename))

time2 = time.perf_counter()
print("NVTX events added, DF saved in {:.2f}s".format(time2 - time1))
print("Done.")
