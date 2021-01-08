import subprocess
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools


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
def fileToDF(logfile_path, pars, debug=False):
    filename_pattern = None  # Get columns values from filename

    if "columns" in pars:
        filename_pattern = pars["filename_pattern"]
        fix_columns = pars["columns"]

    parameters = ["time"]
    if "parameters" in pars:
        parameters = pars["parameters"]

    if debug:
        print "Flatting {}".format(parameters)
    flat_pars = flatten(parameters)
    if debug:
        print "Flat parameters list: {}".format(flat_pars)

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
        if debug:
            print "Remove strings:", remove_str

    logfile_path = logfile_path.strip(" \n")
    logfile = os.path.basename(logfile_path)
    if debug:
        print "Reading", logfile
        print "columns=", fix_columns + flat_pars
    with open(logfile_path, "r") as f:
        fix_values = []

        if filename_pattern is not None:
            ms = filename_pattern.match(logfile)
            if ms:
                for i in range(len(fix_columns)):
                    fix_values.append(ms.group(i + 1))
            else:
                print logfile, "didnt match pattern", filename_pattern.pattern

        df = pd.DataFrame(data=None, columns=fix_columns + flat_pars)
        fix_dict = {}
        for column, value in zip(fix_columns, fix_values):
            fix_dict[column] = value
        if debug:
            print "Fix columns: {}".format(fix_dict)
        # Appending row with fix columns
        df = df.append(fix_dict, ignore_index=True)
        indx = df.index[df.shape[0] - 1]  # DataFrame row numebr (index)
        if debug:
            print "row index = {}".format(indx)
            try:
                display(df.head())
            except:
                print df.head()
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
                            param_values.append(m2.group(j + 1))
                            if debug:
                                print "{} = {}".format(parameters[i][j], m2.group(j + 1))
                            df.loc[indx, parameters[i][j]] = m2.group(j + 1)
                    else:
                        param_values.append(m2.group(1))
                        if debug:
                            print "{} := {}".format(parameters[i], m2.group(1))
                        df.loc[indx, parameters[i]] = m2.group(1)

                    # row +=  param_values
                    if debug:
                        try:
                            display(df.head())
                        except:
                            print df.head()
            if debug and missed_line:
                if len(s) > 6:
                    print "> missed line: /{}/".format(s)

        if missed:
            print "No patterns found in {}".format(logfile_path)
        if debug:
            print "Final DF after parcing all lines in the file:"
            try:
                display(df.head())
            except:
                print df.head()

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
        print "Reading", logfile
        print "columns=", fix_columns + var_columns
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
                    print "Parsed file name to:", fix_values
            else:
                print logfile, "didnt match pattern", filename_pattern.pattern

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
                    print "Appending row:", row
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
                        print logfile, ": b", batch, " l", learn_conv
                    continue
            if batch_conv_pattern is not None:
                m = batch_conv_pattern.match(s)
                if m:
                    batch = int(m.group(1))
                    learn_conv = m.group(2)
                    fix_values = [batch, learn_conv]
                    if debug:
                        print logfile, ": b", batch, " conv", learn_conv

    if debug:
        print df.head()
    return df


# Read file logs from logdir directory
def readLogs(logdir, pars, debug=False, chainer=False):
    filename_pattern = pars["filename_pattern"]

    list_command = "ls -1 " + logdir
    if debug > 0:
        print "Looking in", logdir
    files = []
    proc = subprocess.Popen(list_command.split(" "),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline, b''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir, line)))

    if debug > 0:
        print len(files), "files"
        if debug > 1:
            print files

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
    batch_learn_pattern = pars["batch_learn_pattern"]  # BS and LR read from file pattern
    output_pattern = pars["output_pattern"]  # Read Chainer output pattern
    remove_str = pars["remove_str"]  # Remove strings list for cleaning output lines before parsing
    list_command = "ls -1 " + logdir
    files = []
    proc = subprocess.Popen(list_command.split(" "),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if debug:
        maxfiles = 5
    else:
        maxfiles = 100000000
    for line in iter(proc.stdout.readline, b''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir, line)))

    # if debug: print "files:",files
    df = pd.DataFrame(data=None, columns=["batch", "learn", "epoch", "time"])

    filecounter = 0
    for file in files:
        if debug:
            print file
        df1 = fileToDF_AWS(file, batch_learn_pattern, output_pattern, remove_str, debug)
        if len(df1) > 0:
            df = pd.concat([df, df1], ignore_index=True)
        else:
            print "No data from", file
        filecounter += 1
        if filecounter >= maxfiles:
            return df
    return df


# Read minibatch size and 1st epoch time from files.
# Store in a DataFrame.
def fileToDF_AWS(logfile, batch_learn_pattern, output_pattern, remove_str, debug=False):
    logfile = logfile.strip(" \n")
    filename = os.path.basename(logfile)
    if debug:
        print "FILE", filename
    batch = 0
    learn = 0
    m = re.search(batch_learn_pattern, filename)
    if m:
        batch = int(m.group(1))
        learn = float(m.group(2))
        if debug:
            print "BS,LR:", batch, learn

    with open(logfile, "r") as f:
        lines = f.readlines()
        time = 0
        epoch = 0
        ind = 0  # DataFrame row numebr (index)
        df = pd.DataFrame(data=None, columns=["batch", "learn", "epoch", "time"])
        for line in lines:
            s = line.strip(' \n')
            for rmstr in remove_str:
                s = s.replace(rmstr, "")
            m2 = output_pattern.match(s)
            if m2:
                if debug:
                    print s,
                epoch = int(m2.group(1))
                time = float(m2.group(6))
                if debug:
                    print "BS,LR,epoch,time:", batch, learn, epoch, time
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
        print "Error: Lists cols and types must be of same length. {}!={}".format(len(cols), len(types))
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
        koeff = getMultiplier(time, time_max, time_min, ratio=ratio, degree=degree).astype(int)
        newlist.append(i)
        # Insert value i koeff-1 times
        if koeff > 1:
            if df.iloc[i]["GPU"] == "K80":
                print time, "(s) koeff=", koeff
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
    # print "inverted list size:",len(invert_list)
    if len(invert_list) > testN:
        # Pick testN samples from positions list without rtaining samples randomly
        idx = np.random.choice(len(invert_list), testN, replace=False)
        # Convert a list of positions to a list of indexes in df subset
        idx_test = [invert_list[i] for i in idx]
    else:
        idx_test = invert_list

    # Stratification: replicate samples with lower values (times)
    print GPU,
    if stratify:
        ratio = int(stratify[0])
        degree = int(stratify[1])
        print "before", len(idx_train),
        time_max = df["time"].max()
        time_min = df["time"].min()
        idx_train = Stratify(idx_train, df_tmp, time_min, time_max, ratio=ratio, degree=degree)
        random.shuffle(idx_train)
        print "after", len(idx_train)
    print len(idx_train), "/", len(idx_test)

    samples_df = df_tmp.sort_values(by=["batch"])
    train_df = samples_df.iloc[idx_train]
    test_df = samples_df.iloc[idx_test]
    # print "return:",train_df.shape,test_df.shape
    return (train_df, test_df)

# Returns to DataFrames: with training samples and test samples


def makeTrainingTestDFs(df, n, trainN, testN, stratify=False):
    GPUs = df["GPU"].unique()
    df_train = None
    df_test = None
    for GPU in GPUs:
        train_1, test_1 = pickSamplesForGPU(df, GPU, trainN / n, testN / n, stratify=stratify)
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
    sc0 = axarr[0].scatter(x=Xtrain["batch"].values, y=Ytrain.values, s=2, alpha=0.1)
    sc1 = axarr[1].scatter(x=Xtest["batch"].values, y=Ytest.values, s=2, alpha=.3)
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
        # print text
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
    df_new = pd.DataFrame(columns=group_columns+[series, y])
    for _, group in df.groupby(by=group_columns):
        min_time = np.min(group[y].values)
        fastest_series = [group.iloc[0][v] for v in group_columns] + ["fastest", min_time]
    #     print fastest_series,df_new.shape[0]
        df_new.loc[df_new.shape[0]] = fastest_series 
    
    df_m = pd.concat([df_new, df],axis=0,sort=True)
    df_m.sort_values(by=group_columns, inplace=True)
    return df_m