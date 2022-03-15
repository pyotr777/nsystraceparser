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
from lib import lib3

pp = pprint.PrettyPrinter(indent=4)

ver = '0.17b'

print('Extracting data from a JSON trace file. v.{}'.format(ver))

parser = argparse.ArgumentParser(
    description='NVIDIA NSight Systems JSON trace parser. Extracts time of events.')
parser.add_argument("--file", '-f', default=None, required=True,
                    help="Trace filename to parse.")
parser.add_argument("--events", default=None, required=True, nargs='*',
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
# print(df.head(5))
# print(df.sample(n=5))
print("Read {} rows from {}.".format(df.shape[0], args.file))

# Create detaframes for each event type
# NVTX objects that have NvtxEvent Timestamp
if 'NvtxEvent.Timestamp' in df.columns:
    nvtx = df[df['NvtxEvent.Timestamp'].notna()].dropna(axis=1, how='all')
    # Convert to seconds as displayed in the Nsight System window
    nvtx['start'] = nvtx['NvtxEvent.Timestamp'] * 10e-10
    nvtx['end'] = nvtx['NvtxEvent.EndTimestamp'].fillna(0) * 10e-10
    nvtx['end'] = nvtx['end'].replace(0, np.nan)
    if debug:
        print('NVTX: {}'.format(nvtx['NvtxEvent.Text'].unique()))
else:
    nvtx = None

traces = df[df['TraceProcessEvent.startNs'].notna()].dropna(axis=1, how='all')
traces['start'] = traces['TraceProcessEvent.startNs'] * 10e-10
traces['end'] = traces['TraceProcessEvent.endNs'] * 10e-10
traces['duration'] = (traces['TraceProcessEvent.endNs'] -
                      traces['TraceProcessEvent.startNs']) * 10e-10
if debug:
    print("Traces DF has {} rows.".format(traces.shape[0]))
    print(traces.head(2))
    print("." * 50)
    print(traces.dtypes)
    print("-" * 50)

try:
    sync = df[df['CudaEvent.sync.eventId'].notna()].dropna(axis=1, how='all')
except KeyError as e:
    print(e)
    print("No CudaEvent.sync.eventId events")
    print("Found events:")
    print(df.columns)
    raise (e)

# Convert to seconds
sync['start'] = sync['CudaEvent.startNs'] * 1e-9
sync['end'] = sync['CudaEvent.endNs'] * 1e-9
sync['duration'] = (sync['CudaEvent.endNs'] - sync['CudaEvent.startNs']) * 1e-9
if debug:
    print("Sync DF has {} rows.".format(sync.shape[0]))

# CUDA event kernels objects
kernels = df[df['CudaEvent.correlationId'].notna()].dropna(axis=1, how='all')
# Convert to seconds
kernels['start'] = kernels['CudaEvent.startNs'] * 1e-9
kernels['end'] = kernels['CudaEvent.endNs'] * 1e-9
kernels['duration'] = (kernels['CudaEvent.endNs'] - kernels['CudaEvent.startNs']) * 1e-9

if debug:
    print("Kernels DF has {} rows.".format(kernels.shape[0]))
    print("Kernels has {} columns:".format(kernels.shape[1]))
    print(kernels.columns)

# Names
names = df[df['value'].notna()].dropna(axis=1, how='all')
if debug:
    print("Names DF has {} rows.".format(names.shape[0]))

# Store info about all found events in this DF
final_columns = ['name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side', 'Type']
events = pd.DataFrame(columns=final_columns)
time2 = time.perf_counter()
print("Data type convertions done in {:.2f}s".format(time2 - time1))
time1 = time2
# SEARCH
# Search CUDA kernels and API events matching patterns
matched_kernels_traces = lib3.SearchCUDAKernelsAndAPI(event_name_patterns, names, kernels,
                                                      traces, nvtx, final_columns,
                                                      debug=debug)
if debug and matched_kernels_traces is not None and matched_kernels_traces.shape[0] > 0:
    print('Matched Events:')
    print(matched_kernels_traces)
    print("." * 50)
    print(matched_kernels_traces.dtypes)
    print('-' * 50)

if matched_kernels_traces is not None and matched_kernels_traces.shape[0] > 0:
    # Store info about all found events in this DF
    events = matched_kernels_traces[final_columns].copy()
else:
    print("Found no CUDA kernels matching patterns {}.".format(event_name_patterns))

# Search events matching patterns
event_names_df = names[names.apply(lib3.searchEventPattern,
                                   event_names=event_name_patterns, axis=1)]

df_ = traces.copy()
print("Searching cuDNN, cuBLAS, ... API calls in traces...")
API_events = df_[df_['TraceProcessEvent.name'].isin(event_names_df['id'])].dropna(
    axis=1, how='all').reset_index(drop=True).copy()

if API_events.shape[0] == 0:
    print("Found no API events matching patterns {}.".format(event_name_patterns))

if API_events.shape[0] > 0:
    print("Found {} API events".format(API_events.shape[0]))
    if debug:
        # Check type of fount events, should be 48
        print("\tFound API event types (should be all 48.): {}".format(
            API_events['Type'].unique()))

    # Store API event names
    API_events.loc[:, 'name'] = API_events['TraceProcessEvent.name'].apply(
        lambda x: event_names_df[event_names_df['id'] == x]['value'].values[0])
    # print("Columns\n{}".format(API_events.columns))
    # display(API_events)

    if debug:
        print(API_events.head())
        print("Unique API events: {}".format(','.join(API_events['name'].unique())))

    # Search NVTX regions encompassing API events
    API_events.loc[:, 'NVTX'] = API_events.apply(NVTXforAPIevent, nvtx=nvtx, axis=1)

    # Search CUDA kernels for each API event
    # API events have 3 or more entries:
    # cuDNN/cuBLAS API call, CUDA kernel call(s), CUDA kernel execution(s).
    N = API_events.shape[0]
    for i, row in API_events.iterrows():
        if debug:
            print("{}/{} {}".format(i + 1, N,
                                    ", ".join([str(s) for s in row.loc['start':]])))
        start = row.loc['start']
        end = row.loc['end']
        duration = end - start
        APIname = row['name']
        if row['NVTX'] is None:
            NVTX_s = ''
        else:
            NVTX_s = ','.join(row['NVTX'])
        # Add CPU-side event
        # TODO: Do we need these cuDNN/cuBLAS API events in output (they may not have GPU counterparts)?
        # They also overlap with CUDA API events which have GPU counterparts.
        # Columns: ['name', 'start', 'end', 'duration', 'NVTX', 'corrID','GPU side']
        events.loc[events.shape[0]] = [
            APIname, start, end, duration, NVTX_s, np.nan, False
        ]

        # Search CUDA API events in the time range,
        # store CUDA kernels duration

        df_ = lib3.lookupAPIandKernelsInTimerange(start, end, traces, kernels, names,
                                                  final_columns)
        if df_.shape[0] > 0:
            # Set NVTX
            df_['NVTX'] = NVTX_s
            # Set name for CUDA kernels without names (like memory copy)
            df_.loc[df_['name'].isna(), 'name'] = APIname
            if debug:
                print('{} events for {} nvtx:{} ({:f}-{:f})'.format(
                    df_.shape[0], APIname, NVTX_s, start, end))
                print(df_)
            events = events.append([df_[final_columns]], ignore_index=True)

# NVTX events
if nvtx is not None:
    print("Searching NVTX ...")
    nvtx_events_df = nvtx[nvtx.apply(lib3.searchEventPattern,
                                     event_names=event_name_patterns, debug=extradebug,
                                     axis=1)].reset_index(drop=True).copy()
    if nvtx_events_df.shape[0] == 0:
        print("Found no NVTX events matching patterns {}.".format(event_name_patterns))
    else:
        if args.debug:
            print('Matched {} events'.format(nvtx_events_df.shape[0]))
            print(nvtx_events_df.head())
    try:
        if nvtx_events_df.shape[0] > 0:
            N = nvtx_events_df.shape[0]
            print("Parsing {} NVTX events".format(N))
            for i, nvtx_event in nvtx_events_df.iterrows():
                if debug:
                    print("{}/{}".format(i + 1, N))
                # Find encompassing NVTX ranges
                # import pdb
                # pdb.set_trace()
                start = nvtx_event['start']
                end = nvtx_event['end']
                nvtxranges = nvtx[nvtx['end'].notna()].copy()
                nvtxranges = nvtxranges[nvtxranges['start'] <= start]
                nvtxranges = nvtxranges[nvtxranges['end'] > end]
                nvtx_names = ''
                if nvtxranges.shape[0] > 0:
                    nvtx_names = ','.join(nvtxranges['NvtxEvent.Text'].values)
                    if debug:
                        print('Encompassing NVTX ranges: "{}"'.format(nvtx_names))
                # duration = nvtx_event['end'] - nvtx_event['start']

                # Do not Add NVTX event to events DF because it will be difficult to filter it out
                # to find CPU-time of API events.
                # final_columns =  'name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side'
                # events.loc[events.shape[0]] = [
                #     nvtx_event['NvtxEvent.Text'], nvtx_event['start'],
                #     nvtx_event['end'], duration, nvtx_names, np.nan, False
                # ]

                # Find CUDA kernel time (start, end, duration) for each NVTX event

                events_in_nvtx = lib3.lookupAPIandKernelsInTimerange(
                    start, end, traces, kernels, names, final_columns)
                nvtx_ = nvtx_event['NvtxEvent.Text']
                # Append encompassing NVTX ranges
                if len(nvtx_names) > 0:
                    nvtx_ = nvtx_names + "," + nvtx_
                events_in_nvtx['NVTX'] = nvtx_
                if debug:
                    print('CUDA Kernels for NVTX range "{}":'.format(
                        nvtx_event['NvtxEvent.Text']))
                    pp.pprint(events_in_nvtx.head())
                    # Check no name kernels
                    noname = events_in_nvtx.loc[events_in_nvtx['name'].isna(),
                                                ['name', 'NVTX', 'corrID']]
                    if noname.shape[0] > 0:
                        print("Events without name:")
                        pp.pprint(noname)
                        noname_corrids = noname['corrID'].unique()
                        df_ = events_in_nvtx[events_in_nvtx['corrID'].isin(
                            noname_corrids)].sort_values('corrID', ignore_index=True)
                        print("Grouped by corrID")
                        pp.pprint(df_)

                events = events.append(events_in_nvtx, ignore_index=True)

    except NameError:
        print()
        print('No NVTX events in the trace.')

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

print("Done.")
