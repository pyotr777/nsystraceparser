#!/usr/bin/env python3

# Library functions for Python3
# (C) 2019 Peter Bryzgalov @ CHITEC

import subprocess
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import six


# Print text in color
def message(s, col=112):
    print("\033[38;5;{}m{}\033[0m".format(col, s))


# Flatten list of lists
def flatten(multilist):
    flatlist = []
    if isinstance(multilist, list):
        for l in multilist:
            if isinstance(l, list):
                flatlist += flatten(l)
            else:
                flatlist.append(l)
    else:
        flatlist = multilist
    return flatlist


# Parse filename for fix_columns values,
# parse each line from the file, searching for parameters' values.
# Parameters are a list of lists, or list of strings, or a mixed list.
# Each member of parameters list corresponds to the search pattern from output_patterns.
# output_patterns is a list of the same length as the top level of parameters list.
# If any output_pattern (member of output patterns) designed to match several groups in one line,
# values of matched groups with be saved to columns with names from corresponding item in parameters list;
# corresponding item of parameters list must be also a list with the length matching the number of groups in
# the output_pattern.
# Return a DataFrame with one line only with the parsed values.
def fileToDF(logfile_path, pars, debug=0):
    filename_pattern = None  # Get columns values from filename

    if "columns" in pars:
        filename_pattern = pars["filename_pattern"]
        fix_columns = pars["columns"]

    parameters = ["time"]
    if "parameters" in pars:
        parameters = pars["parameters"]

    if debug > 0:
        print("Flattening {}".format(parameters))
    flat_pars = flatten(parameters)
    if debug > 0:
        print("Flat parameters list: {}".format(flat_pars))

    output_patterns = None
    if "output_patterns" in pars:
        output_patterns = pars["output_patterns"]
    else:
        # Backward compatibility
        output_patterns = [pars["output_pattern"]]

    remove_str = None
    if "remove_str" in pars:
        remove_str = pars["remove_str"]
        if type(remove_str) is not list:
            remove_str = [remove_str]
        if debug > 0:
            print("Remove strings:", remove_str)

    logfile_path = logfile_path.strip(" \n")
    logfile = os.path.basename(logfile_path)
    if debug > 0:
        print("-" * 12)
        message("Reading {}".format(logfile))
        print("columns=", fix_columns + flat_pars)
    important_columns = ['machine', 'batch', 'run', 'epoch', 'time']
    with open(logfile_path, "r") as f:
        fix_values = []

        if filename_pattern is not None:
            ms = filename_pattern.match(logfile)
            if ms:
                for i in range(len(fix_columns)):
                    fix_values.append(ms.group(i + 1))
            else:
                print(logfile, "didnt match pattern", filename_pattern.pattern)

        df = pd.DataFrame(data=None, columns=fix_columns + flat_pars)
        fix_dict = {}
        for column, value in zip(fix_columns, fix_values):
            fix_dict[column] = value
        if debug > 0:
            print("Fix columns: {}".format(fix_dict))
        # Appending row with fix columns
        df = df.append(fix_dict, ignore_index=True)
        indx = df.index[df.shape[0] - 1]  # DataFrame row numebr (index)
        if debug > 0:
            print("row index = {}".format(indx))
            try:
                print(df.head())
            except:
                print(df.head())
        lines = f.readlines()
        missed = True  # Missed all patterns
        for line in lines:
            s = line.strip(' \n')
            if remove_str:
                for rmstr in remove_str:
                    s = s.replace(rmstr, "")
            missed_line = True
            for i, output_pattern in enumerate(output_patterns):
                m2 = output_pattern.search(s)
                if m2:
                    missed = False
                    missed_line = False
                    param_values = []
                    if isinstance(parameters[i], list):
                        for j in range(len(parameters[i])):
                            val = m2.group(j + 1)
                            param_values.append(val)
                            if debug > 0:
                                print("{} = {}".format(parameters[i][j], val))
                            # If have multiple epochs make new line for each epoch starting from the second
                            if parameters[i][j] == 'epoch':
                                # If more than 1 epoch in one log file,
                                # copy previous string for epoch > 1, and set new values only for epoch and time.
                                if debug > 1:
                                    print('Epoch {}, epoch in the row: {}'.
                                          format(val, df.loc[indx]['epoch']))
                                    print('Make new row? {}'.format(
                                        not pd.isna(df.loc[indx]['epoch'])))
                                if not pd.isna(df.loc[indx]['epoch']):
                                    indx += 1
                                    df.loc[indx] = df.loc[indx - 1]
                                df.loc[indx, 'epoch'] = val
                            else:
                                df.loc[indx,
                                       parameters[i][j]] = m2.group(j + 1)

                    else:
                        param_values.append(m2.group(1))
                        if debug > 0:
                            print("{} := {}".format(parameters[i],
                                                    m2.group(1)))
                        df.loc[indx, parameters[i]] = m2.group(1)

                    # row +=  param_values
                    if debug > 0:
                        try:
                            print(df[important_columns].head())
                        except:
                            pass
            if debug > 0 and missed_line:
                if len(s) > 6:
                    print("> missed line: /{}/".format(s))

        if missed:
            print("No patterns found in {}".format(logfile_path))
        if debug > 0:
            print("Final DF after parcing all lines in the file:")
            try:
                print(df.head())
            except:
                pass

    return df


# Read minibatch size and 1st epoch time from files.
# Store in a DataFrame.
def ChainerfileToDF(logfile_path, pars, debug=False):
    batch_learn_pattern = None
    batch_conv_pattern = None
    filename_pattern = None  # Get columns values from filename

    if "batch_learn_pattern" in pars:
        batch_learn_pattern = pars["batch_learn_pattern"]
        fix_columns = ["batch", "learn"]

    if "batch_conv_pattern" in pars:
        batch_conv_pattern = pars["batch_conv_pattern"]
        fix_columns = ["batch", "conv"]

    if "columns" in pars:
        filename_pattern = pars["filename_pattern"]
        fix_columns = pars["columns"]

    var_groups = [1, 6]  # Column numbers in log files to parse epoch and time
    if "var_groups" in pars:
        var_groups = pars["var_groups"]

    var_columns = ["epoch", "time"]
    output_pattern = pars["output_pattern"]
    remove_str = pars["remove_str"]

    logfile_path = logfile_path.strip(" \n")
    logfile = os.path.basename(logfile_path)
    if debug:
        print("Reading", logfile)
        print("columns=", fix_columns + var_columns)
    with open(logfile_path, "r") as f:
        batch = 0
        learn_conv = 0
        fix_values = []
        time = 0
        epoch = 0
        ind = 0  # DataFrame row numebr (index)

        if filename_pattern is not None:
            ms = filename_pattern.match(logfile)
            if ms:
                for i in range(len(fix_columns)):
                    fix_values.append(ms.group(i + 1))
                if debug:
                    print("Parsed file name to:", fix_values)
            else:
                print(logfile, "didnt match pattern", filename_pattern.pattern)

        df = pd.DataFrame(data=None, columns=fix_columns + var_columns)
        row = []
        lines = f.readlines()
        for line in lines:
            s = line.strip(' \n')
            if type(remove_str) is not list:
                remove_str = [remove_str]
            for rmstr in remove_str:
                s = s.replace(rmstr, "")
            m2 = output_pattern.match(s)
            if m2:
                epoch = int(m2.group(var_groups[0]))
                time = float(m2.group(var_groups[1]))
                row = fix_values + [epoch, time]
                if debug:
                    print("Appending row:", row)
                df.loc[ind] = row
                ind += 1
                continue
            if batch_learn_pattern is not None:
                m = batch_learn_pattern.match(s)
                if m:
                    batch = int(m.group(1))
                    learn_conv = float(m.group(2))
                    fix_values = [batch, learn_conv]
                    if debug:
                        print(logfile, ": b", batch, " l", learn_conv)
                    continue
            if batch_conv_pattern is not None:
                m = batch_conv_pattern.match(s)
                if m:
                    batch = int(m.group(1))
                    learn_conv = m.group(2)
                    fix_values = [batch, learn_conv]
                    if debug:
                        print(logfile, ": b", batch, " conv", learn_conv)

    if debug:
        print(df.head())
    return df


# Read file logs from logdir directory
def readLogs(logdir, pars, debug=0, chainer=False, maxfiles=1):
    filename_pattern = pars["filename_pattern"]

    list_command = "ls -1 " + logdir
    if debug > 0:
        print("Looking in", logdir)
    files = []
    proc = subprocess.Popen(list_command.split(" "),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            encoding='utf8')
    for line in iter(proc.stdout.readline, ''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir, line)))

    print('{} files in {}'.format(len(files), logdir))
    if debug > 0:
        if maxfiles is not None and maxfiles != 0:
            files = files[:maxfiles]
        print(files)

    df = None

    if chainer:
        for file in files:
            df1 = ChainerfileToDF(file, pars=pars, debug=debug)
            if len(df1) > 0:
                if df is None:
                    df = df1
                else:
                    df = pd.concat([df, df1], ignore_index=True)
    else:
        for file in files:
            df1 = fileToDF(file, pars=pars, debug=debug)
            if len(df1) > 0:
                if df is None:
                    df = df1
                else:
                    df = pd.concat([df, df1], ignore_index=True)
    return df


# Read file logs from logdir directory
# Chainer log files from AWS has extra garbage to be removed
def readLogsAWS(logdir, pars, debug=False):
    filename_pattern = pars["filename_pattern"]  # Log files file names pattern
    batch_learn_pattern = pars[
        "batch_learn_pattern"]  # BS and LR read from file pattern
    output_pattern = pars["output_pattern"]  # Read Chainer output pattern
    remove_str = pars[
        "remove_str"]  # Remove strings list for cleaning output lines before parsing
    list_command = "ls -1 " + logdir
    files = []
    proc = subprocess.Popen(list_command.split(" "),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if debug:
        maxfiles = 5
    else:
        maxfiles = 100000000
    for line in iter(proc.stdout.readline, b''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir, line)))

    # if debug: print("files:",files)
    df = pd.DataFrame(data=None, columns=["batch", "learn", "epoch", "time"])

    filecounter = 0
    for file in files:
        if debug:
            print(file)
        df1 = fileToDF_AWS(file, batch_learn_pattern, output_pattern,
                           remove_str, debug)
        if len(df1) > 0:
            df = pd.concat([df, df1], ignore_index=True)
        else:
            print("No data from", file)
        filecounter += 1
        if filecounter >= maxfiles:
            return df
    return df


# Read minibatch size and 1st epoch time from files.
# Store in a DataFrame.
def fileToDF_AWS(logfile,
                 batch_learn_pattern,
                 output_pattern,
                 remove_str,
                 debug=False):
    logfile = logfile.strip(" \n")
    filename = os.path.basename(logfile)
    if debug:
        print("FILE", filename)
    batch = 0
    learn = 0
    m = re.search(batch_learn_pattern, filename)
    if m:
        batch = int(m.group(1))
        learn = float(m.group(2))
        if debug:
            print("BS,LR:", batch, learn)

    with open(logfile, "r") as f:
        lines = f.readlines()
        time = 0
        epoch = 0
        ind = 0  # DataFrame row numebr (index)
        df = pd.DataFrame(data=None,
                          columns=["batch", "learn", "epoch", "time"])
        for line in lines:
            s = line.strip(' \n')
            for rmstr in remove_str:
                s = s.replace(rmstr, "")
            m2 = output_pattern.match(s)
            if m2:
                if debug:
                    six.print_(s, end="")
                epoch = int(m2.group(1))
                time = float(m2.group(6))
                if debug:
                    print("BS,LR,epoch,time:", batch, learn, epoch, time)
                df.loc[ind] = [batch, learn, epoch, time]
                ind += 1

    return df


# Convert df columns to the given types
# df - DataFrame
# cols - list of columns, or list of lists
# type - type or list of types
def convertColumnTypes(df, cols, types):
    if type(cols) is not list:
        cols = [cols]
    if type(types) is not list:
        types = [types]

    if len(cols) != len(types):
        print("Error: Lists cols and types must be of same length. {}!={}".
              format(len(cols), len(types)))
        return df

    for col, ctype in zip(cols, types):
        if type(col) is not list:
            col = [col]

        for column in col:
            df[column] = df[column].astype(ctype)
    return df


# Returns replication count as a function of x "importance" - smaller values more important
# used in replicating important training samples.
# ratio - number of replications for most important samples,
# degree - "stepness" of the function curve
def getMultiplier(x, xmax, xmin, ratio=5, degree=2):
    x = x - xmin
    mxx = xmax - xmin
    y = (mxx - x) / (mxx + x)
    y = (y**degree) * (ratio - 1) + 1
    return np.round(y).astype(int)


# Return list of n indexes uniformly spreaded in 0-l range
def pickSampleIndexes(l, n):
    x = []
    step = float(l) / float(n)
    for i in range(l):
        pos = int(round(step * i))
        if pos < l:
            x.append(pos)
    return x


# Replicate samples proportionally to their inverted value (time):
# Samples with small values get replicated more.
def Stratify(idx, df, time_min, time_max, ratio=5, degree=2):
    newlist = []
    for i in idx:
        time = df.iloc[i]["time"]
        koeff = getMultiplier(time,
                              time_max,
                              time_min,
                              ratio=ratio,
                              degree=degree).astype(int)
        newlist.append(i)
        # Insert value i koeff-1 times
        if koeff > 1:
            if df.iloc[i]["GPU"] == "K80":
                print(time, "(s) koeff=", koeff)
            l = [i] * (koeff - 1)
            newlist = newlist + l
    return newlist


# Pick equally spaced N samples from Dataframe df where "GPU" column is GPU
def pickSamplesForGPU(df, GPU, trainN, testN, stratify=False):
    # Use equally spaced samples for training set
    df_tmp = df[df["GPU"] == GPU]
    l = len(df_tmp.index)
    idx_train = pickSampleIndexes(l, trainN)
    # idx_train is a list of positions in df subset (rows for specific GPU model)
    # Exclude training set positions from list of row positions in df subset
    invert_list = [i for i in range(l) if i not in idx_train]
    # print("inverted list size:",len(invert_list))
    if len(invert_list) > testN:
        # Pick testN samples from positions list without rtaining samples randomly
        idx = np.random.choice(len(invert_list), testN, replace=False)
        # Convert a list of positions to a list of indexes in df subset
        idx_test = [invert_list[i] for i in idx]
    else:
        idx_test = invert_list

    # Stratification: replicate samples with lower values (times)
    six.print_(GPU, end="")
    if stratify:
        ratio = int(stratify[0])
        degree = int(stratify[1])
        six.print_("before", len(idx_train), end="")
        time_max = df["time"].max()
        time_min = df["time"].min()
        idx_train = Stratify(idx_train,
                             df_tmp,
                             time_min,
                             time_max,
                             ratio=ratio,
                             degree=degree)
        random.shuffle(idx_train)
        print("after", len(idx_train))
    print(len(idx_train), "/", len(idx_test))

    samples_df = df_tmp.sort_values(by=["batch"])
    train_df = samples_df.iloc[idx_train]
    test_df = samples_df.iloc[idx_test]
    # print("return:",train_df.shape,test_df.shape)
    return (train_df, test_df)


# Returns to DataFrames: with training samples and test samples


def makeTrainingTestDFs(df, n, trainN, testN, stratify=False):
    GPUs = df["GPU"].unique()
    df_train = None
    df_test = None
    for GPU in GPUs:
        train_1, test_1 = pickSamplesForGPU(df,
                                            GPU,
                                            trainN / n,
                                            testN / n,
                                            stratify=stratify)
        if df_train is None:
            df_train = train_1
        else:
            df_train = pd.merge(df_train, train_1, how="outer")

        if df_test is None:
            df_test = test_1
        else:
            df_test = pd.merge(df_test, test_1, how="outer")
    return (df_train, df_test)


# Plot two plots with training samples and test samples
def plotTrainTestSamples(Xtrain, Ytrain, Xtest, Ytest):
    f, axarr = plt.subplots(1, 2, sharex=True, figsize=(12, 3))
    sc0 = axarr[0].scatter(x=Xtrain["batch"].values,
                           y=Ytrain.values,
                           s=2,
                           alpha=0.1)
    sc1 = axarr[1].scatter(x=Xtest["batch"].values,
                           y=Ytest.values,
                           s=2,
                           alpha=.3)
    axarr[0].set_title("training set")
    axarr[1].set_title("test set")
    axarr[0].grid(ls=":", alpha=0.1)
    axarr[1].grid(ls=":", alpha=0.1)
    plt.show()


# Mean Absolute Percentage Error
# Renamed PercentageError function
def MAPE(y, h):
    h = np.array(h)
    y = np.array(y)
    err = np.mean(np.abs((y - h) / y)) * 100
    return err


# Returns Percentage Error
# For compatibility with older notebooks


def PercentageError(y, h):
    return MAPE(y, h)


# Plot prediction line
# df - Dataframe with ALL samples
# idx - indexes of samples from the test set
def plotPredictions1(model, df, df_test, title, features):
    no_batch_features = features[1:]
    # no_batch_features.remove("batch")
    df_tmp = pd.DataFrame(columns=features)
    pad = 15
    bmin = df_test["batch"].min() - pad
    bmax = df_test["batch"].max() + pad
    x_ = np.arange(bmin, bmax, 5)
    architectures = df_test["CUDA cap"].unique()
    architectures = sorted(architectures, key=str, reverse=True)
    #height = len(architectures) * 3
    fig, ax = plt.subplots(len(architectures), 1, sharex=True, figsize=(9, 9))
    ax[0].set_title(title)
    for i in range(len(architectures)):
        CUDA_cap = architectures[i]
        GPU = df[df["CUDA cap"] == CUDA_cap]["GPU"].iloc[0]
        add = df[df["CUDA cap"] == CUDA_cap][no_batch_features].iloc[0].values
        for j in range(len(x_)):
            df_tmp.loc[j] = np.insert(add, 0, x_[j])
        y_ = model.predict(df_tmp)
        #         x_ = df_test[df_test["GPU"]==GPU]["batch"].values
        #         y_ = model.predict(df_test[df_test["GPU"]==GPU][features].values)
        ax[i].plot(x_, y_, c="r", label="prediction " + GPU)

        # Plot test samples
        Xc = df_test[df_test["GPU"] == GPU][features].values
        Yc = df_test[df_test["GPU"] == GPU]["time"].values
        Htest = model.predict(Xc)
        X = df_test[df_test["GPU"] == GPU]["batch"]
        ax[i].scatter(X, Yc, s=1, alpha=.5, label="test samples")
        MPE = "MPE={:.5f}".format(PercentageError(Yc, Htest))
        # print(text)
        ax[i].set_ylabel("time (s)")
        ax[i].grid(ls=":", alpha=0.3)
        ax[i].legend()
        ax[i].text(1.01, 0.9, MPE, transform=ax[i].transAxes, size=12)
    ax[-1].set_xlabel("batch size")
    fig.show()


# Select lower values from Y per X from multiple series
# group_columns - column(s) to split df into groups (not includes series column)
# series_column  - one column that identify series,
# y - one column with numeric values to choose the lowest.
def getLowestFromSeries(df, group_columns, series, y):
    df_new = pd.DataFrame(columns=group_columns + [series, y])
    for _, group in df.groupby(by=group_columns):
        min_time = np.min(group[y].values)
        fastest_series = [group.iloc[0][v]
                          for v in group_columns] + ["fastest", min_time]
        df_new.loc[df_new.shape[0]] = fastest_series

    df_m = pd.concat([df_new, df], axis=0, sort=True)
    df_m.sort_values(by=group_columns, inplace=True)
    return df_m


# Convert Pandas to formatted string
# Formats:
# compact - no extra spaces or new lines,
# dict - to python dictionary,
# full - multiline string with alighed values.
def series2string(ser, format='compact', debug=False):
    arr = []
    if debug:
        print('Series:\n', ser.to_string())
    if format == 'compact':
        for ind, val in ser.items():
            val = val[0]  # Get value from a short series
            arr.append('{}:{}'.format(ind, val))
        if debug:
            print('Compact format:')
            print(' '.join(arr))
        return ' '.join(arr)
    if format == 'full':
        for ind, val in ser.iteritems():
            val = val[0]  # Get value from a short series
            arr.append('{:20}:    {}'.format(ind, val))
        if debug:
            print('Meta in full format:')
            print('\n'.join(arr))
        return '\n'.join(arr)
    if format == 'dict':
        return ser.to_dict()
    else:
        print('Wrong format {}. Supported values: compact, dict and full.'.
              format(format))
    return None


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
def lookupAPIandKernelsInTimerange(start,
                                   end,
                                   traces,
                                   kernels,
                                   names,
                                   final_columns,
                                   debug=False):
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
        else:
            # CudaEvent.kernel.shortName is None for all GPU-side kernels
            # Create an empty column "name", it's used in merging with results later.
            kernel_events.loc[:, 'name'] = None
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

            if debug:
                print("Merged tables with names:")
                pp.pprint(m_.head())
        except Exception as e:
            print(e)
            if debug:
                import pdb
                pdb.set_trace()
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
