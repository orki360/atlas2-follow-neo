#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"
exec python3.12 setup_env.py
