"""Start the ngrok CLI as an independent background process.

Reads NGROK_DOMAIN and NGROK_AUTHTOKEN from .env / environment.
Writes the ngrok PID to /tmp/ngrok-swe.pid and stdout log to /tmp/ngrok-swe.log.
"""
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

ngrok_bin = shutil.which("ngrok")
if not ngrok_bin:
    print("ERROR: ngrok CLI not found — install from https://ngrok.com/download")
    sys.exit(1)

domain = os.getenv("NGROK_DOMAIN", "")
token = os.getenv("NGROK_AUTHTOKEN", "")

args = [ngrok_bin, "http", "8000", "--log=stdout", "--log-format=json"]
if domain:
    args.append(f"--domain={domain}")
if token:
    args.append(f"--authtoken={token}")

proc = subprocess.Popen(args, stdout=open("/tmp/ngrok-swe.log", "w"), stderr=subprocess.STDOUT)
with open("/tmp/ngrok-swe.pid", "w") as f:
    f.write(str(proc.pid))

print(f"  ngrok PID: {proc.pid}")
