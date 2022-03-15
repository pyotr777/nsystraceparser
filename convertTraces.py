#!/usr/bin/env python3

# Run nsight tracing series with a varying parameter.

import multigpuexec
import os
import argparse
import ast
import datetime
import subprocess
from subprocess import Popen

print('Converting Nsight Systems traces to JSON. v0.01.')

parser = argparse.ArgumentParser(
    description='Convert all NVIDIA Nsight Systems traces .qdrep -> .json in a directory.'
)

parser.add_argument("--dir", default='.', help="Folder to save traces to.")
args = parser.parse_args()

# Read traces
list_command = "ls -1 " + args.dir
files = []
proc = subprocess.Popen(list_command.split(" "), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, encoding='utf8')
for line in iter(proc.stdout.readline, ''):
    line = line.strip(" \n")
    i = line.find('.qdrep')
    if i > 1:
        files.append(os.path.join(args.dir, line))

print('Have {} trace files.'.format(len(files)))

for file in files:
    # Strip extention
    filename = '.'.join(os.path.basename(file).split('.')[:-1])
    # append path
    filename = os.path.join(args.dir, filename)

    # Convert trace to JSON
    command = "nsys export --type json -o {fname}.json --force-overwrite true --separate-strings true {fname}.qdrep".format(
        fname=filename)
    print("Converting report {}:".format(filename))
    print(command)
    p = subprocess.run(command.split(' '), stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                       bufsize=0, shell=False)
    if p.returncode != 0:
        print('Convertion failed.')
        print(p.stdout)
        print(p.stderr)

print("No more tasks to run. Logs are in {}.".format(args.dir))
