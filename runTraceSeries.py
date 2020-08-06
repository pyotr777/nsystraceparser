#!/usr/bin/env python3

# Run nsight tracing series with a varying parameter.

import multigpuexec
import os
import argparse
import ast
import datetime
import subprocess
from subprocess import Popen

print('Running Trace Series with a variable parameter. v0.03.')

parser = argparse.ArgumentParser(
    description=
    'Run series of NVIDIA Nsight Systems tracer for commands with a varying parameter.')

parser.add_argument("--iter", type=int, default=10,
                    help="Number of iterations to profile.")
parser.add_argument("--gpu", default='0',
                    help="GPU number, which GPU to use for running trace commands.")
parser.add_argument("--dir", default='.', help="Folder to save traces to.")
parser.add_argument('--date', default=None, help='Set date for the logs path.')
parser.add_argument("--host", default=None, help="Host name")
parser.add_argument(
    "--output", '-o', default='nsys_trace_$p',
    help="Traces file name pattern. '$p' will be replaced with parameter value.")
parser.add_argument("--parameter", '-p', type=int, default=None, nargs='*',
                    help="Space-separated list of mini-batch sizes.")
args = parser.parse_args()

# Change command
# Use $p placeholder for the varying parameter
command_template = 'python3 ../mlbenchmarks/pytorch/examples/imagenet/imagenet.py'\
' --arch resnet50 -e 1 --iter {} -b $p --workers 3 --nvtx --imnet /host/Imagenet/images/'.format(
    args.iter)
print('Command to execute:')
print(command_template)
# Change parameter values
if args.parameter is None:
    parameters = [5, 6, 7, 8, 9, 10, 12, 15] + list(range(20, 131, 10))
else:
    parameters = args.parameter
print("Parameter values: {}".format(parameters))

host = multigpuexec.getHostname()
if args.host:
    host = args.host
date = datetime.datetime.today().strftime('%Y%m%d')
if args.date:
    date = args.date
    print("Using date {}".format(date))
logdir = os.path.join("logs", host, "traces", date)
logbase = args.dir
logdir = os.path.join(logbase, logdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Saving logs to", logdir)

gpu = args.gpu
print("GPU: {}".format(gpu))

for parameter in parameters:
    filename = args.output.replace('$p', str(parameter))
    # Tracing command
    report_file = os.path.join(logdir, filename)
    trace_command = 'nsys profile -s cpu -t cuda,cudnn,nvtx,cublas -o {}'.format(
        report_file)

    logfilename = 'stdout_{}.log'.format(parameter)
    logfile = os.path.join(logdir, logfilename)
    command = command_template.replace('$p', str(parameter))
    if os.path.isfile(logfile):
        print("file", logfile, "exists.")
        continue
    # Run tracing
    with open(logfile, "w+") as f:
        print('Getting GPU info')
        gpu_info = multigpuexec.getGPUinfo(gpu, path='/host/mlbenchmarks/')
        print('Getting CPU info')
        cpu_info = multigpuexec.getCPUinfo()
        f.write("command:{}\n".format(command))
        f.write("GPU{}: {}".format(gpu, gpu_info))
        f.write("{}".format(cpu_info))
        # Set GPU for execution with env var CUDA_VISIBLE_DEVICES
        my_env = os.environ.copy()
        my_env[b"CUDA_VISIBLE_DEVICES"] = str(gpu)
        my_env[b"NVIDIA_VISIBLE_DEVICES"] = str(gpu)
        exec_command = trace_command + ' ' + command
        multigpuexec.message(exec_command)
        p = subprocess.run(exec_command.split(' '), stdout=f, stderr=subprocess.STDOUT,
                           bufsize=0, env=my_env)

    # Convert trace to JSON
    command = "nsys export --type json -o {qdrep}.json --separate-strings true {qdrep}.qdrep".format(
        qdrep=report_file)
    print("Converting report:")
    print(command)
    # p = subprocess.run(command.split(' '), stdin=subprocess.PIPE,
    #                    stderr=subprocess.PIPE, check=False, universal_newlines=True)
    p = subprocess.run(command.split(' '), stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                       bufsize=0, shell=False)
    # if len(p.stderr) > 0:
    #     print(p.stderr, end='')

print("No more tasks to run. Logs are in {}/{}.json files.".format(logdir, args.output))
