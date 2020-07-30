# NVIDIA Nsight Systems Traces Parser

A set of Python scripts for:

a. Taking traces of multiple runs of the target application with one varying parameter,  
b. Converting traces to JSON format,  
c. Extracting from each trace the (GPU) time of API calls (cuDNN, cuBLAS, etc.) and NVTX regions,  
d. Aggregating extracted times across multiple traces with the varying parameter.


### runTraceSeries

Runs a series of the target application runs with one varying parameter, makes nsys trace for each run and converts traces to JSON format.

```
usage: runTraceSeries.py [-h] [--iter ITER] [--gpu GPU] [--dir DIR]
                         [--date DATE] [--host HOST] [--output OUTPUT]
                         [--parameter [PARAMETER [PARAMETER ...]]]

Run series of NVIDIA Nsight Systems tracer for commands with a varying
parameter.

optional arguments:
  -h, --help            show this help message and exit
  --iter ITER           Number of iterations to profile.
  --gpu GPU             GPU number, which GPU to use for running trace
                        commands.
  --dir DIR             Folder to save traces to.
  --date DATE           Set date for the logs path.
  --host HOST           Host name
  --output OUTPUT, -o OUTPUT
                        Traces file name pattern. '$p' will be replaced with
                        parameter value.
  --parameter [PARAMETER [PARAMETER ...]], -p [PARAMETER [PARAMETER ...]]
                        Space-separated list of mini-batch sizes.
```


### convertTraces

Converts existing nsys traces to JSON format.

```
usage: convertTraces.py [-h] [--dir DIR]

Convert all NVIDIA Nsight Systems traces .qdrep -> .json in a directory.

optional arguments:
  -h, --help  show this help message and exit
  --dir DIR   Folder to save traces to.
```


### parseOneTrace

Extracts API event times from a trace file in JSON format.

```
usage: parseOneTrace.py [-h] --file FILE --events [EVENTS [EVENTS ...]]

NVIDIA NSight Systems JSON trace parser. Extracts time of events.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Trace filename to parse.
  --events [EVENTS [EVENTS ...]]
                        Event name patterns. Multiple space-separated values
                        possible.
```

