"""Run an uploaded job script as a child process and stream its output.

Running each step as its own process frees GPU memory between steps (train, then evaluate). Environment variables:

  JOB_SCRIPT   path of the uploaded script on the GPU machine
  JOB_ARGS     its command-line arguments, as one shell-quoted string
"""

import os
import shlex
import subprocess
import sys

script = os.environ["JOB_SCRIPT"]
cmd = [sys.executable, "-u", script, *shlex.split(os.environ.get("JOB_ARGS", ""))]
print(f"[launch] {' '.join(cmd)[:300]}", flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
for line in proc.stdout:
    print(line, end="", flush=True)
code = proc.wait()
print(f"[launch] exit {code}", flush=True)
if code:
    raise RuntimeError(f"{script} exited with {code}")
