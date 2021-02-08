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
import pprint
pp = pprint.PrettyPrinter(indent=4)

ver = '0.16f'

print('Extracting data from a JSON trace file. v.{}'.format(ver))

parser = argparse.ArgumentParser(
    description=
    'NVIDIA NSight Systems JSON trace parser. Extracts time of events.')
parser.add_argument("--file",
                    '-f',
                    default=None,
                    required=True,
                    help="Trace filename to parse.")
parser.add_argument(
    "--events",
    default=None,
    required=True,
    nargs='*',
    help="Event name patterns. Multiple space-separated values possible.")
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()
debug = args.debug
event_name_patterns = args.events
extradebug = False
if debug:
    print(datetime.datetime.now())


# Get all rows from DF with the given correlation ID.
# Search in all df columns with 'correlationId' in the name.
def LookupCorrelationID(corrId, df):  # nvtx, cuda, kernels, sync):
    corrid_columns = [
        c for c in df.columns if c.lower().find('correlationid') >= 0
    ]
    df_ = df[df[corrid_columns].eq(corrId).any(1)].copy().dropna(axis=1,
                                                                 how='all')
    return df_


# Search all trace events (CPU side CUDA API) and corresponding CUDA kernels (GPU side)
# matching name patterns.
# No cuDNN or cuBLAS API events should be found here (they do not have correlation IDs).
# Return DF with columns 'start', 'end', 'duration', 'name', 'NVTX', 'GPU side'
def SearchCUDAKernelsAndAPI(patterns,
                            names,
                            kernels,
                            traces,
                            nvtx,
                            final_columns,
                            debug=False):
    event_names_df = names[names.apply(searchEventPattern,
                                       event_names=patterns,
                                       axis=1)]
    if event_names_df.shape[0] == 0:
        print("No names found for patterns '{}'".format(','.join(patterns)))
        print("Searching again with consize syntax...")
        event_names_df = None
        for pattern in patterns:
            df_ = names[names['value'].str.match(pattern)].copy()
            if debug:
                print("Found {} names for pattern '{}'.".format(
                    df_.shape[0], pattern))
                print(df_)
            if event_names_df is None:
                event_names_df = df_
            else:
                event_names_df = event_names_df.append(df_, ignore_index=True)

        if event_names_df.shape[0] == 0:
            print("No names found for patterns '{}'".format(
                ','.join(patterns)))
            return None
    ids = event_names_df['id'].values
    matched_traces = pd.DataFrame()
    # Search CUDA kernels by shortName field (ids correspond to kernel names)
    matched_kernels = kernels[kernels['CudaEvent.kernel.shortName'].isin(
        ids)].dropna(axis=1, how="all").copy()
    if debug:
        print("Found {} events matching name patterns '{}'.".format(
            event_names_df.shape[0], ','.join(patterns)))
        print(event_names_df)
        print("Found {} CUDA kernels matching events by names.".format(
            matched_kernels.shape[0]))
    if matched_kernels.shape[0] > 0:
        # Found some matching CUDA kernels
        matched_kernels = matched_kernels[[
            'Type', 'CudaEvent.kernel.shortName', 'CudaEvent.startNs',
            'CudaEvent.endNs', 'CudaEvent.correlationId', 'start', 'end',
            'duration'
        ]]
        matched_kernels.loc[:, 'name'] = matched_kernels[
            'CudaEvent.kernel.shortName'].apply(lambda s: event_names_df[
                event_names_df['id'] == s]['value'].values[0])
        matched_kernels.loc[:, 'NVTX'] = np.nan
        matched_kernels.loc[:, 'GPU side'] = True
        matched_kernels.rename(columns={'CudaEvent.correlationId': 'corrID'},
                               inplace=True)
        # Search corresponding CUDA API (traces)
        matched_traces = traces[traces['TraceProcessEvent.correlationId'].isin(
            matched_kernels['corrID'].unique())].dropna(axis=1,
                                                        how="all").copy()

        if matched_traces.shape[0] > 0:
            # Found some matching traces (CPU-side) events
            # TraceProcessEvent.name is not importnat (all same?)
            # using corresponding CUDA kernel names.
            matched_traces.loc[:, 'name'] = matched_traces[
                'TraceProcessEvent.correlationId'].apply(
                    lambda s: matched_kernels[matched_kernels['corrID'] == s][
                        'name'].values[0])
            matched_traces.loc[:,
                               'NVTX'] = matched_traces.apply(NVTXforAPIevent,
                                                              nvtx=nvtx,
                                                              axis=1)
            matched_traces.loc[:, 'GPU side'] = False
            matched_traces.rename(
                columns={'TraceProcessEvent.correlationId': 'corrID'},
                inplace=True)

    if matched_kernels.shape[0] > 0 and matched_traces.shape[0] > 0:
        # Concat API events (traces) and CUDA kernels
        try:
            merged = pd.concat([
                matched_kernels[final_columns], matched_traces[final_columns]
            ],
                               ignore_index=True)
        except Exception as e:
            print(e)
            print("Traces head")
            print(matched_traces.head())
            print("Kernels head")
            print(matched_kernels.head())
        return merged
    elif matched_kernels.shape[0] > 0:
        return matched_kernels
    elif matched_traces.shape[0] > 0:
        return matched_traces
    else:
        return None


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
# final_columns =  'name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side', 'Type'
def lookupAPIandKernelsInTimerange(start, end, traces, kernels, names,
                                   final_columns):
    # Lookup traces (API) events in the given range
    # import pdb
    # pdb.set_trace()
    startdf = traces[traces['start'] >= start]
    rangedf = startdf[startdf['end'] <= end].copy()
    if rangedf.shape[0] == 0:
        print("No events found for time range {}-{}.".format(start, end))
        return None

    # This always evals to cudaLaunchKernel_v7000 for some type of kernels:
    rangedf.loc[:, 'name'] = rangedf['TraceProcessEvent.name'].apply(
        lambda s: names[names['id'] == s]['value'].values[0])
    # However, just in case no CUDA kernels found, for now use this dummy name anyway.
    # If kernels found, than use correlating CUDA kernel name.
    rangedf.loc[:, 'GPU side'] = False
    rangedf.loc[:, 'NVTX'] = 'TBD'  # Set later outside of the function
    rangedf.rename(columns={'TraceProcessEvent.correlationId': 'corrID'},
                   inplace=True)
    if debug:
        print(
            "lookupAPIandKernelsInTimerange: In time range {}-{} found {} trace events."
            .format(start, end, rangedf.shape[0]))
        print(rangedf.dropna(axis=1, how="all").head())
        noname = rangedf[rangedf['name'].isna()]
        if noname.shape[0] > 0:
            print("{} no name trace events in lookupAPIandKernelsInTimerange.".
                  format(noname.shape[0]))
            print(noname)
        # print("Columns: {}".format(",".join(rangedf.columns)))

    # Store results in the DF
    results = rangedf[final_columns].copy()

    kernel_events = kernels[kernels['CudaEvent.correlationId'].isin(
        results['corrID'].unique())].copy()
    if kernel_events.shape[0] == 0:
        print("Found no CUDA kernels for corrIDs: {}".format(",".join(
            results['corrID'].unique())))
    else:
        kernel_events.rename(columns={'CudaEvent.correlationId': 'corrID'},
                             inplace=True)
        if kernel_events.loc[kernel_events['CudaEvent.kernel.shortName'].notna(
        )].shape[0] > 0:
            kernel_events['CudaEvent.kernel.shortName'] = kernel_events[
                'CudaEvent.kernel.shortName'].fillna(-1).astype(int).replace(
                    -1, np.nan)
            try:
                kernel_events.loc[
                    kernel_events['CudaEvent.kernel.shortName'].notna(),
                    'name'] = kernel_events[
                        kernel_events['CudaEvent.kernel.shortName'].notna(
                        )]['CudaEvent.kernel.shortName'].apply(lambda s: names[
                            names['id'] == s]['value'].values[0])
            except IndexError as e:
                print("Error in lookupAPIandKernelsInTimerange")
                print(e)
                print("Test names DF")
                rows = kernel_events[kernel_events[
                    'CudaEvent.kernel.shortName'].notna()].shape[0]
                print("Have {} valid rows".format(rows))
                if rows > 0:
                    for i, s in kernel_events[kernel_events[
                            'CudaEvent.kernel.shortName'].notna()].iterrows():
                        name = int(s.loc['CudaEvent.kernel.shortName'])
                        test_name_values = names[names['id'] == name]['value']
                        if len(test_name_values) == 0:
                            print("no names for event\n v v v v")
                            pp.pprint(s)
                            print(" ^ ^ ^ ^")

                else:
                    print(kernel_events)
                sys.exit(1)

        kernel_events.loc[:, 'GPU side'] = True
        kernel_events.loc[:,
                          'NVTX'] = 'TBD'  # Set later outside of the function
        if debug:
            print("Found {} CUDA kernels:".format(kernel_events.shape[0]))
            print(kernel_events.dropna(axis=1, how="all").head())

        # Make a merged table with names and corrIDs only
        try:
            k_ = kernel_events.loc[kernel_events['corrID'] != 0.,
                                   ['corrID', 'name']]
            m_ = pd.merge(results[['corrID', 'name']],
                          k_,
                          on=['corrID'],
                          suffixes=["", "_kernel"]).fillna('')

            # import pdb
            # pdb.set_trace()
            if debug:
                print("Merged tables with names:")
                pp.pprint(m_)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Set missing event names:
        # for API events use kernel names,
        # for kernel events use API event names.
        try:

            def setMissingNames(s, namedf):
                badnames = ['', 'cudaLaunchKernel_v7000'
                            ]  # These are the names to be replaced
                corrid = s.loc['corrID']
                name = s.loc['name']
                # Check in badnames
                if name in badnames:
                    availnames = list(
                        namedf.loc[namedf['corrID'] == corrid,
                                   ['name', 'name_kernel']].values.flatten())
                    goodnames = list(set(availnames) - set(badnames))
                    return goodnames[0]
                else:
                    return s.loc['name']

            results.loc[results['corrID'] != 0, 'name'] = results[
                results['corrID'] != 0].fillna('').apply(setMissingNames,
                                                         namedf=m_,
                                                         axis=1)
            kernel_events.loc[kernel_events['corrID'] != 0,
                              'name'] = kernel_events[kernel_events['corrID']
                                                      != 0].fillna('').apply(
                                                          setMissingNames,
                                                          namedf=m_,
                                                          axis=1)
            if debug:
                print("Testing names")
                # Test bad names
                badnames = ['', 'cudaLaunchKernel_v7000']
                badnamesresults = results[results['name'].fillna('').isin(
                    badnames)]
                if badnamesresults.shape[0] > 0:
                    print("No good names found for:")
                    print(badnamesresults)

        except Exception as e:
            print("Cannot set names in lookupAPIandKernelsInTimerange")
            print(e)

        results = results.append(kernel_events[final_columns])

    return results


# Find NVTX event which encompasses given trace event
def NVTXforAPIevent(trace_event, nvtx, debug=False):
    if nvtx is None:
        return None

    # Start and end in seconds
    try:
        start = trace_event.loc['TraceProcessEvent.startNs'] * 10e-10
        end = trace_event.loc['TraceProcessEvent.endNs'] * 10e-10
    except KeyError as e:
        print(e)
        print('columns: {}'.format(trace_event))
    if debug:
        print('{} - {}'.format(start, end))
    # Search NVTX object encompassing events
    nvtxranges = nvtx[nvtx['end'].notna()].copy()
    nvtxranges = nvtxranges[nvtxranges['start'] <= start]
    nvtxranges = nvtxranges[nvtxranges['end'] >= end]
    names = nvtxranges['NvtxEvent.Text'].values
    return names


# Return True if the row value contains any of event name patterns
def searchEventPattern(row, event_names=None, debug=False):
    s = None
    if 'value' in row.index:
        s = row.loc['value']
    elif 'NvtxEvent.Text' in row.index:
        s = row.loc['NvtxEvent.Text']
    else:
        print('Can search only Names and NVTX dataframes.')
        return False
    for pattern in event_names:
        m = re.match(pattern, s, re.I)
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
tmpfile = '.'.join(args.file.split('.')[:-1]) + "_raw.csv"
if os.path.exists(tmpfile):
    print("Reading from raw CSV file {}".format(tmpfile))
    df = pd.read_csv(tmpfile)
else:
    print("Reading {}...".format(args.file))
    if not os.path.exists(args.file):
        print("File {} not found.".format(args.file))
    data = [json.loads(line) for line in open(args.file, 'r')]

    df = json_normalize(data)
    df.to_csv(tmpfile, index=False)
    print("Saved DF to raw CSV file {}".format(tmpfile))

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
    print(traces.head())
    print("." * 50)
    print(traces.dtypes)
    print("-" * 50)

sync = df[df['CudaEvent.sync.eventId'].notna()].dropna(axis=1, how='all')
# Convert to seconds
sync['start'] = sync['CudaEvent.startNs'] * 10e-10
sync['end'] = sync['CudaEvent.endNs'] * 10e-10
sync['duration'] = (sync['CudaEvent.endNs'] -
                    sync['CudaEvent.startNs']) * 10e-10
if debug:
    print("Sync DF has {} rows.".format(sync.shape[0]))

# CUDA event kernels objects
kernels = df[df['CudaEvent.correlationId'].notna()].dropna(axis=1, how='all')
# Convert to seconds
kernels['start'] = kernels['CudaEvent.startNs'] * 10e-10
kernels['end'] = kernels['CudaEvent.endNs'] * 10e-10
kernels['duration'] = (kernels['CudaEvent.endNs'] -
                       kernels['CudaEvent.startNs']) * 10e-10

if debug:
    print("Kernels DF has {} rows.".format(kernels.shape[0]))
    print("Kernels has {} columns:".format(kernels.shape[1]))
    print(kernels.columns)

# Names
names = df[df['value'].notna()].dropna(axis=1, how='all')
if debug:
    print("Names DF has {} rows.".format(names.shape[0]))

# Store info about all found events in this DF
final_columns = [
    'name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side', 'Type'
]
events = pd.DataFrame(columns=final_columns)

# SEARCH
# Search CUDA kernels and API events matching patterns
matched_kernels_traces = SearchCUDAKernelsAndAPI(event_name_patterns,
                                                 names,
                                                 kernels,
                                                 traces,
                                                 nvtx,
                                                 final_columns,
                                                 debug=debug)
if debug and matched_kernels_traces is not None and matched_kernels_traces.shape[
        0] > 0:
    print('Matched Events:')
    print(matched_kernels_traces)
    print("." * 50)
    print(matched_kernels_traces.dtypes)
    print('-' * 50)

if matched_kernels_traces is not None and matched_kernels_traces.shape[0] > 0:
    # Store info about all found events in this DF
    events = matched_kernels_traces[final_columns].copy()
else:
    print("Found no CUDA kernels matching patterns {}.".format(
        event_name_patterns))

# Search events matching patterns
event_names_df = names[names.apply(searchEventPattern,
                                   event_names=event_name_patterns,
                                   axis=1)]
df_ = traces.copy()
print("Searching cuDNN, cuBLAS, ... API calls in traces...")
API_events = df_[df_['TraceProcessEvent.name'].isin(
    event_names_df['id'])].dropna(axis=1,
                                  how='all').reset_index(drop=True).copy()

if API_events.shape[0] == 0:
    print("Found no API events matching patterns {}.".format(
        event_name_patterns))

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
        print("Unique API events: {}".format(','.join(
            API_events['name'].unique())))

    # Search NVTX regions encompassing API events
    API_events.loc[:, 'NVTX'] = API_events.apply(NVTXforAPIevent,
                                                 nvtx=nvtx,
                                                 axis=1)

    # Search CUDA kernels for each API event
    # API events have 3 or more entries:
    # cuDNN/cuBLAS API call, CUDA kernel call(s), CUDA kernel execution(s).
    N = API_events.shape[0]
    for i, row in API_events.iterrows():
        if debug:
            print("{}/{} {}".format(
                i + 1, N, ", ".join([str(s) for s in row.loc['start':]])))
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
        df_ = lookupAPIandKernelsInTimerange(start, end, traces, kernels,
                                             names, final_columns)
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

            # for _, event in df_.iterrows():
            #     # CPU-side event
            #     events.loc[events.shape[0]] = [
            #         event['kernel'], event['api_start'], event['api_end'],
            #         event['api_end'] - event['api_start'], NVTX_s,
            #         event['correlationId'], False
            #     ]
            #     # GPU-side event
            #     events.loc[events.shape[0]] = [
            #         event['kernel'], event['start'], event['end'],
            #         event['duration'], NVTX_s, event['correlationId'], True
            #     ]

# NVTX events
if nvtx is not None:
    print("Searching NVTX ...")
    nvtx_events_df = nvtx[nvtx.apply(searchEventPattern,
                                     event_names=event_name_patterns,
                                     debug=extradebug,
                                     axis=1)].reset_index(drop=True).copy()
    if nvtx_events_df.shape[0] == 0:
        print("Found no NVTX events matching patterns {}.".format(
            event_name_patterns))
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
                        print('Encompassing NVTX ranges: "{}"'.format(
                            nvtx_names))
                # duration = nvtx_event['end'] - nvtx_event['start']

                # Do not Add NVTX event to events DF because it will be difficult to filter it out
                # to find CPU-time of API events.
                # final_columns =  'name', 'start', 'end', 'duration', 'NVTX', 'corrID', 'GPU side'
                # events.loc[events.shape[0]] = [
                #     nvtx_event['NvtxEvent.Text'], nvtx_event['start'],
                #     nvtx_event['end'], duration, nvtx_names, np.nan, False
                # ]

                # Find CUDA kernel time (start, end, duration) for each NVTX event

                events_in_nvtx = lookupAPIandKernelsInTimerange(
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
                            noname_corrids)].sort_values('corrID',
                                                         ignore_index=True)
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