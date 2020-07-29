#!/usr/bin/env python3

# Extract data from one JSON trace file.
# 1. Find all events matching patterns from the --events option.
# 2. Find  Find all API (CPU side) events with the two names. Store in DF API_events.
# 3. Find NVTX ranges encompassing each API event. Add 'NVTX' column to API_events.
# 4. Find CUDA kernels (GPU side) for each API event. Lookup CUDA calls (CPU side) within
#    the time range of each API event, than lookup CUDA kernels (GPU side) by correlation IDs.
# 5. Aggregate time of all CUDA kernels in each iteration. Return aggregated time per iteration.

import os
import argparse
import pandas as pd
import json
from pandas import json_normalize
import numpy as np
import re
import sys

print('Extracting data from a JSON trace file. v.0.03.')

parser = argparse.ArgumentParser(
    description='NVIDIA NSight Systems JSON trace parser. Extracts time of events.')
parser.add_argument("--file", '-f', default=None, required=True,
                    help="Trace filename to parse.")
parser.add_argument("--events", default=None, required=True, nargs='*',
                    help="Event name patterns. Multiple space-separated values possible.")
args = parser.parse_args()


# Get all rows from DF with the given correlation ID.
# Search in all df columns with 'correlationId' in the name.
def LookupCorrelationID(corrId, df):  # nvtx, cuda, kernels, sync):
    corrid_columns = [c for c in df.columns if c.lower().find('correlationid') >= 0]
    dfcorr = None
    for c in corrid_columns:
        df_ = df[df[c] == corrId]
        if dfcorr is None:
            dfcorr = df_
        else:
            dfcorr = dfcorr.append(df_)
    return dfcorr.dropna(axis=1, how='all')


# Convert columns StartNs and EndNs to
# start and end in seconds.
def convertStartEndTimes(df):
    df_ = df.copy()
    df_['start'] = None
    df_['end'] = None
    start_cols = [c for c in df.columns if c.lower().find('startns') >= 0]
    end_cols = [c for c in df.columns if c.lower().find('endns') >= 0]
    for c in start_cols:
        rows = df_[c].notna()
        df_.loc[rows, 'start'] = df_.loc[rows, c] * 10e-10
    for c in end_cols:
        rows = df_[c].notna()
        df_.loc[rows, 'end'] = df_.loc[rows, c] * 10e-10
    return df_


# Get CUDA kernel names for events with the given correlationIDs
def LookupNamebyCorrID(corrid, df, names):
    dfcorr = LookupCorrelationID(corrid, df)
    if dfcorr.shape[0] == 0:
        return []
    namestrings = []
    if 'CudaEvent.kernel.shortName' in dfcorr.columns:
        if dfcorr['CudaEvent.kernel.shortName'].notna().any():
            shortnames = dfcorr['CudaEvent.kernel.shortName']
            shortnames = shortnames[shortnames.notna()].values
            for ID in shortnames:
                try:
                    n = int(ID)
                except:
                    print('Cannot convert {} to int.'.format(n))
                    continue
                namestrings.append(names[names['id'] == n]['value'].values[0])
    return namestrings


# Return rows that contain string
def searchRowsContaining(s, df):
    mask = df.applymap(lambda x: s.lower() in str(x).lower())
    df_ = df[mask.any(axis=1)]
    return df_


# Search events from df within the time range.
# DF must have 'start' and 'end' columns.
def lookupTimeRange(start, end, df):
    startdf = df[df['start'] >= start]
    rangedf = startdf[startdf['end'] <= end]
    return rangedf


# Combine trace evenets within time range and cuda kernels lookup
def lookupAPIandKernelsInTimerange(start, end, traces, kernels, names):
    # Lookup traces (API) events in the given range
    startdf = traces[traces['start'] >= start]
    rangedf = startdf[startdf['end'] <= end]
    # Store results in the DF
    results = pd.DataFrame(columns=[
        'correlationId', 'api_start', 'api_end', 'kernel', 'start', 'end', 'duration'
    ])

    for i, row in rangedf.iterrows():
        # Get correlation ID from the trace event
        corrID = row['TraceProcessEvent.correlationId']
        if corrID == 0:
            continue
        # Get CUDA kernel by correlation ID
        kernel_event = LookupCorrelationID(corrID, kernels)
        if kernel_event is None or kernel_event.shape[0] == 0:
            # No kernels for trace event with the corrID
            continue
        # Get the name of the CUDA kernel
        name = LookupNamebyCorrID(corrID, kernels, names)
        # Append to results DF
        results.loc[results.shape[0]] = [
            corrID, row['start'], row['end'], name[0], kernel_event['start'].values[0],
            kernel_event['end'].values[0], kernel_event['duration'].values[0]
        ]
    return results


# Find NVTX event which encompasses given trace event
def NVTXforAPIevent(trace_event):
    # Start and end in seconds
    start = trace_event.loc['TraceProcessEvent.startNs'] * 10e-10
    end = trace_event.loc['TraceProcessEvent.endNs'] * 10e-10
    # Search NVTX object encompassing events
    nvtxranges = nvtx[nvtx['end'].notna()].copy()
    nvtxranges = nvtxranges[nvtxranges['start'] <= start]
    nvtxranges = nvtxranges[nvtxranges['end'] >= end]
    names = nvtxranges['NvtxEvent.Text'].values
    return names


# Return True if the row value contains any of event name patterns
def searchEventPattern(row, event_names=None, debug=False):
    s = row.loc['value']
    for pattern in event_names:
        m = re.search(pattern, s, re.I)
        if m is not None:
            return True
        else:
            if debug:
                print("{} not found in {}".format(pattern, s))
    return False


# Parse an array of nvtx range names for iteration number
def GetIterationNumber(nvtx_arr):
    nvtx_name = [n for n in nvtx_arr if 'iteration' in n.lower()]
    if len(nvtx_name) == 0:
        return None
    nvtx_name = nvtx_name[0]  # Convert list to string
    s = nvtx_name.replace('Iteration ', '')
    try:
        i = int(s)
    except:
        print('Cannot convert {} to int'.format(s))
        return None
    return i


# Read Trace file
print("Reading", args.file)
if not os.path.exists(args.file):
    print("File {} not found.".format(args.file))
data = [json.loads(line) for line in open(args.file, 'r')]

df = json_normalize(data)
int_columns = [
    'NvtxEvent.Timestamp', 'NvtxEvent.EndTimestamp', 'CudaEvent.startNs',
    'CudaEvent.endNs', 'CudaEvent.correlationId', 'CudaEvent.sync.eventId',
    'TraceProcessEvent.correlationId', 'TraceProcessEvent.name',
    'TraceProcessEvent.startNs', 'TraceProcessEvent.endNs', 'id'
]
for c in int_columns:
    df[c] = df[c].fillna(-1).astype(int).replace(-1, np.nan)

df.dropna(axis=1, how='all')
# print(df.head(5))
# print(df.sample(n=5))
print("Read {} rows from {}.".format(df.shape[0], args.file))

# Create detaframes for each event type
# NVTX objects that have NvtxEvent Timestamp
nvtx = df[df['NvtxEvent.Timestamp'].notna()].dropna(axis=1, how='all')
# Convert to seconds as displayed in the Nsight System window
nvtx['start'] = nvtx['NvtxEvent.Timestamp'] * 10e-10
nvtx['end'] = nvtx['NvtxEvent.EndTimestamp'].fillna(0) * 10e-10
nvtx['end'] = nvtx['end'].replace(0, np.nan)
print('NVTX')
print(nvtx.head())

traces = df[df['TraceProcessEvent.startNs'].notna()].dropna(axis=1, how='all')
traces['start'] = traces['TraceProcessEvent.startNs'] * 10e-10
traces['end'] = traces['TraceProcessEvent.endNs'] * 10e-10
traces['duration'] = (traces['TraceProcessEvent.endNs'] -
                      traces['TraceProcessEvent.startNs']) * 10e-10

sync = df[df['CudaEvent.sync.eventId'].notna()].dropna(axis=1, how='all')
# Convert to seconds
sync['start'] = sync['CudaEvent.startNs'] * 10e-10
sync['end'] = sync['CudaEvent.endNs'] * 10e-10
sync['duration'] = (sync['CudaEvent.endNs'] - sync['CudaEvent.startNs']) * 10e-10

# CUDA event kernels objects
kernels = df[df['CudaEvent.kernel.shortName'].notna()].dropna(axis=1, how='all')
# Convert to seconds
kernels['start'] = kernels['CudaEvent.startNs'] * 10e-10
kernels['end'] = kernels['CudaEvent.endNs'] * 10e-10
kernels['duration'] = (kernels['CudaEvent.endNs'] - kernels['CudaEvent.startNs']) * 10e-10

# Names
names = df[df['value'].notna()].dropna(axis=1, how='all')

# print("Names")
# print(names.head())

# Search events matching patterns
event_name_patterns = args.events
event_names_df = names[names.apply(searchEventPattern, event_names=event_name_patterns,
                                   axis=1)]
print('Matched Events:')
print(event_names_df)

if event_names_df.shape[0] == 0:
    print("Found no events matching patterns {}.".format(args.events))

# Search trace events (cuDNN, cuBLAS API events, CPU side)
df_ = traces.copy()
API_events = df_[df_['TraceProcessEvent.name'].isin(event_names_df['id'])].dropna(
    axis=1, how='all')
print("Found {} API events".format(API_events.shape[0]))
if API_events.shape[0] == 0:
    sys.exit(0)

# Store API event names
API_events['name'] = API_events['TraceProcessEvent.name'].apply(
    lambda x: event_names_df[event_names_df['id'] == x]['value'].values[0])
# print("Columns\n{}".format(API_events.columns))
# display(API_events)
print("Unique API events:")
print(API_events['name'].unique())

# Search NVTX reagons encompassing API events
API_events['NVTX'] = API_events.apply(NVTXforAPIevent, axis=1)

print("API with NVTX:")
print(API_events.head(2))

# Search CUDA API calls for each API event
cudakernels = None
for _, row in API_events.iterrows():
    start = row.loc['start']
    end = row.loc['end']
    APIname = row['name']
    NVTX_arr = row['NVTX']
    NVTX_s = ','.join(row['NVTX'])
    df_ = lookupAPIandKernelsInTimerange(start, end, traces, kernels, names)
    # print('{}kernels for {:} {} ({:.4f}-{:.4f})'.format(df_.shape[0], APIname, NVTX_s,
    #                                                     start, end))
    df_['API event'] = APIname
    df_['NVTX_arr'] = df_.apply(lambda x: NVTX_arr, axis=1)
    df_['NVTX'] = NVTX_s

    #     display(df_)
    if cudakernels is None:
        cudakernels = df_
    else:
        cudakernels = cudakernels.append(df_, ignore_index=True)

cudakernels['iteration'] = cudakernels['NVTX_arr'].apply(GetIterationNumber)
# If NVTX ranges do not include Iteration, 'iteration' column will have None-s.
cudakernels.dropna(axis=1, how='all', inplace=True)
if 'iteration' in cudakernels.columns:
    print('Have iterations in NVTX.')
    # print(cudakernels['iterations'])
    use_columns = ['duration', 'NVTX', 'API event', 'iteration']
else:
    print('No iterations data.')
    use_columns = ['duration', 'NVTX', 'API event']
group_by_columns = list(set(use_columns) -
                        set(['duration']))  # Group by all columns except duration
agg_kernels = cudakernels[use_columns].groupby(group_by_columns, as_index=False).sum()
print(agg_kernels)

directory = os.path.dirname(args.file)
filename = ('.').join(os.path.basename(
    args.file).split('.')[:-1])  # Filename without extension
filename = filename + '.csv'
filename = os.path.join(directory, filename)
agg_kernels.to_csv(filename, index=False)
print('Saved to {}.'.format(filename))
