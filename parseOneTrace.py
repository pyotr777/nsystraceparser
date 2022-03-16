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
import sqlite3
import numpy as np
import re
import sys
from multigpuexec import message
import datetime
import time
import pprint
from lib import lib3

pp = pprint.PrettyPrinter(indent=4)

ver = '1.02a'

print('Extracting data from a SQlight trace file. v.{}'.format(ver))


# Finds encompassing NVTX ranges for the event.
# r is a row from the kernels DF with start and end fields.
# Returns list of matching NVTX ranges.
def NVTXfilterRange(r, nvtx):
    start = r['start']
    end = r['end']
    nvtx_ = nvtx[nvtx['start'] <= start]
    nvtx_ = nvtx_[nvtx_['end'] >= end]
    ranges = nvtx_['text'].unique()
    return ranges


# Replaces name ID with the name from names DF
# Input: r is a row with ID to replace, col is the column name with name ID to be replaced,
# names is the names DF.
# Returns the event name (to replace name ID)
def replaceIdWithName(r, col, names):
    if pd.isna(r[col]):
        return None
    vals = None
    try:
        vals = names[names['id'] == r[col]]['value']
        if vals is None:
            return r[col]
        elif len(vals) == 0:
            return r[col]
        else:
            return vals.values[0]
    except Exception as e:
        print('!' * 25)
        print(e)
        print(r)
        print(col)
        print("vals: ", vals)
        raise Exception('error!')


def main():
    parser = argparse.ArgumentParser(
        description='NVIDIA NSight Systems JSON trace parser. Extracts time of events.')
    parser.add_argument("--file", '-f', default=None, required=True,
                        help="nsys trace file to parse.")
    parser.add_argument(
        "--event-filters", default=None, required=False, nargs='*', help=
        "Event (kernels etc.) name patterns. Multiple space-separated values possible.")
    parser.add_argument("--nvtx-filters", default=None, required=False, nargs='*',
                        help="Patterns for filtering NVTX")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    debug = args.debug

    if debug:
        # List tables
        with sqlite3.connect(args.file) as con:
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
        tables = [l[0] for l in tables]
        print(tables)

    # Mark: Read from SQlite Tables to DF names, nvtx, kernels, traces
    with sqlite3.connect(args.file) as con:
        names = pd.read_sql_query("SELECT * FROM StringIds;", con)
        nvtx = pd.read_sql_query("SELECT * FROM NVTX_EVENTS;", con)
        kernels = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL;", con)
        traces = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME;", con)

    # Mark: reformat dataframes

    # NVTX
    nvtx = nvtx.dropna(how='all', axis=1)
    int_columns = ['eventType', 'globalTid', 'domainId']
    int_columns = [c for c in int_columns if c in nvtx.columns]
    for c in int_columns:
        nvtx[c] = nvtx[c].fillna(-1).astype(int).replace(-1, np.nan)
    # Convert nanosecs to secs
    nvtx['start'] = nvtx['start'] * 1e-9
    nvtx['end'] = nvtx['end'].fillna(0) * 1e-9
    nvtx['end'] = nvtx['end'].replace(0, np.nan)

    # traces (API calls on the host side)
    traces['start'] = traces['start'] * 1e-9
    traces['end'] = traces['end'] * 1e-9
    traces.loc[:, 'duration'] = (traces['end'] - traces['start'])

    # kernels
    kernels = kernels.dropna(how='all', axis=1)
    # Convert to seconds
    kernels['start'] = kernels['start'] * 1e-9
    kernels['end'] = kernels['end'] * 1e-9

    # Mark: filter kernel names

    # Find IDs for events with the names matching filters
    # Search for ids in names DF

    if args.event_filters is not None:
        events = None
        for pattern in args.event_filters:
            events_ = names[names['value'].str.match(pattern, flags=re.I)]
            if events is None:
                events = events_
            else:
                events = pd.concat([events, events_], ignore_index=True).drop_duplicates()
        event_ids = events['id'].unique()

        # Search for kernels matching patterns
        matching_kernels = None
        for i in event_ids:
            df_ = kernels[(kernels['demangledName'] == i) | (kernels['shortName'] == i) |
                          (kernels['mangledName'] == i)]
            if df_.size == 0:
                continue
            if matching_kernels is None:
                matching_kernels = df_
            else:
                matching_kernels = pd.concat([matching_kernels, df_], ignore_index=True)
        kernels = matching_kernels

    # Find traces with matching correlation IDs
    matching_corrIDs = sorted(kernels['correlationId'].unique())
    matching_traces = traces[traces['correlationId'].isin(matching_corrIDs)]

    # Mark: filter nvtx

    if args.nvtx_filters is not None:
        matching_nvtx = None
        for nvtx_filter in args.nvtx_filters:
            print(nvtx_filter)
            nvtx_ = nvtx[nvtx['text'].str.match(nvtx_filter, flags=re.I)]
        if matching_nvtx is None:
            matching_nvtx = nvtx_
        else:
            matching_nvtx = pd.concat([matching_nvtx, nvtx_])
        nvtx = matching_nvtx

    # Mark: merge traces with NVTX

    traces.loc[:, 'nvtx'] = traces.apply(lambda x: NVTXfilterRange(x, nvtx), axis=1)

    # Mark: merge traces with kernels

    traces_columns = ['start', 'end', 'nameId', 'nvtx', 'correlationId']
    kernel_columns = [
        'start', 'end', 'deviceId', 'correlationId', 'demangledName', 'shortName',
        'mangledName'
    ]
    eventsDF = pd.merge(traces[traces_columns], kernels[kernel_columns], how='outer',
                        on='correlationId', suffixes=['', '_gpu'])

    # Replace name IDs with names in eventsDF
    name_columns = ['nameId', 'demangledName', 'shortName', 'mangledName']
    for col in name_columns:
        eventsDF[col] = eventsDF.apply(lambda x: replaceIdWithName(x, col, names), axis=1)
    eventsDF = eventsDF.rename(columns={'nameId': 'CUDA API'})

    # Mark: save eventsDF and NVTX DF to CSV files
    nvtx_file_ext = '_nvtx.csv'
    events_file_ext = '_evnt.csv'
    nvtxfile = os.path.splitext(args.file)[0] + nvtx_file_ext
    eventsfile = os.path.splitext(args.file)[0] + events_file_ext
    nvtx.to_csv(nvtxfile, index=False)
    print(f"Saved nvtx to {nvtxfile}")
    eventsDF.to_csv(eventsfile, index=False)
    print(f"Saved events to {eventsfile}")

    print("All done.")


if __name__ == '__main__':
    main()