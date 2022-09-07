#!/usr/bin/env python3

# Run nsight tracing series with a varying parameter.

import multigpuexec
import os
import argparse
import ast
import datetime
import subprocess
from subprocess import Popen

print('Running Trace Series with a variable parameter. v0.11d.')

parser = argparse.ArgumentParser(
    description=
    'Run series of NVIDIA Nsight Systems tracer for commands with a varying parameter.')

parser.add_argument("--iter", type=int, default=10,
                    help="Number of iterations to profile.")
parser.add_argument("--gpu", default='0',
                    help="GPU number, which GPU to use for running trace commands.")
parser.add_argument("--dir", default='.', help="Folder to save traces to.")
parser.add_argument("--model", default='resnet18', help="CNN model (architecture)")
parser.add_argument("--suffix", default=None, help="Traces directory name suffix.")
parser.add_argument('--date', default=None, help='Set date for the logs path.')
parser.add_argument("--host", default=None, help="Host name")
parser.add_argument("--imagenet", default='/host/imagenet10/images',
                    help="Path to the Imagenet images directory.")
parser.add_argument('--dataset', default='imagenet', help='cifar/imagenet')
parser.add_argument(
    "--output", '-o', default='nsys_trace_#p',
    help="Traces file name pattern. '#p' will be replaced with parameter value.")
parser.add_argument("--parameter", '-p', type=int, default=None, nargs='*',
                    help="Space-separated list of mini-batch sizes.")
parser.add_argument('--script', default='examples/cputimer.py',
                    help="Pytorch script for training on samples.")
parser.add_argument('--basedir', default='/host/mlbench/',
                    help='Absolute path to mlbench dirsctory')
parser.add_argument('--traceext', default='nsys-rep', help="Trace file extension")
args = parser.parse_args()
# Change command
# Use $p placeholder for the varying parameter
datasetname = ''
if args.dataset == "auto":
    # Select Dataset based on the CNN model
    if "vgg" in args.model:
        datasetname = "cifar"
    else:
        datasetname = "imagenet"
else:
    datasetname = args.dataset

more_options = ''
if datasetname == "imagenet":
    more_options += '--imnet {}'.format(args.imagenet)
elif 'cifar' in datasetname:
    # Do not use cifar100 - it will only change path to logs but cifar100 will be used anyway
    datasetname = 'cifar'

command_template = "python3 {} --dataset {} -e 1 --iter {} -b #p --workers 5 --nvtx --arch {} {}".format(
    os.path.join(args.basedir, 'pytorch', args.script), datasetname, args.iter,
    args.model, more_options)

print('Command to execute:')
print(command_template)
# Change parameter values
if args.parameter is None:
    parameters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15] + list(range(20, 101, 10))
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
if args.suffix is None:
    dirname = date
else:
    dirname = "{}{}".format(date, args.suffix)
logdir = os.path.join("logs", host, "traces", args.model, dirname)
logbase = args.dir
logdir = os.path.join(logbase, logdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Saving logs to", logdir)

gpu = args.gpu
print("GPU: {}".format(gpu))

for parameter in parameters:
    filename = args.output.replace('#p', str(parameter))
    print("Trace filename without extension: {}".format(filename))
    # Tracing command
    report_file = os.path.join(logdir, filename)
    trace_command = 'nsys profile -t cuda,cudnn,osrt,cublas,nvtx -s none -o {}'.format(
        report_file)

    logfilename = '{}_stdout.log'.format(filename)
    logfile = os.path.join(logdir, logfilename)
    command = command_template.replace('#p', str(parameter))
    if os.path.isfile(logfile):
        print("file", logfile, "exists.")
        continue
    # Run tracing
    with open(logfile, "w+") as f:
        print('Getting GPU info')
        gpu_info = multigpuexec.getGPUinfo(gpu, path=args.basedir)
        # Set GPU for execution with env var CUDA_VISIBLE_DEVICES
        my_env = os.environ.copy()
        my_env[b"CUDA_VISIBLE_DEVICES"] = str(gpu)
        my_env[b"NVIDIA_VISIBLE_DEVICES"] = str(gpu)
        p = subprocess.run(
            'nvidia-smi --query-gpu=gpu_name,count,index --format=csv'.split(' '),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=my_env)
        print(p.stdout.decode('utf-8'))
        print(p.stderr.decode('utf-8'))

        print('Getting CPU info')
        cpu_info = multigpuexec.getCPUinfo()
        f.write("command:{}\n".format(command))
        f.write("GPU{}: {}".format(gpu, gpu_info))
        f.write("{}".format(cpu_info))

        exec_command = trace_command + ' ' + command
        multigpuexec.message(exec_command.split(' '))
        p = subprocess.run(exec_command.split(' '), stdout=f, stderr=subprocess.STDOUT,
                           bufsize=0, shell=False, env=my_env)

    # Convert trace to JSON
    command = "nsys export -f true -t sqlite --separate-strings=true {report}.{ext}".format(
        report=filename, ext=args.traceext)
    print("Converting report:")
    print(command)
    # p = subprocess.run(command.split(' '), stdin=subprocess.PIPE,
    #                    stderr=subprocess.PIPE, check=False, universal_newlines=True)
    p = subprocess.run(command.split(' '), stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                       bufsize=0, shell=False, check=False, cwd=logdir)
    # if len(p.stderr) > 0:
    #     multigpuexec.message(p.stderr, 160)

print("No more tasks to run. Logs are in {}.".format(logdir))
