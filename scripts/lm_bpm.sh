#!/bin/bash
# Run this script from the repository base path

cd "$(dirname "$0")"
cd ..
export PYTHONPATH=$PWD:$PYTHONPATH

# insert path to correct environment
python3 scripts/viewer_p06.py
