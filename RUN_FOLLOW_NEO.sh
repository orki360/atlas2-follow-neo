#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"
exec .venv/bin/python run_lab.py "$@"
