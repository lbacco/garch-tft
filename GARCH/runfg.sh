#!/bin/bash

# The path to your Python script
PYTHON_SCRIPT_PATH="./GARCHCSVs.py"
# Step 1: Setup environment.
echo
echo Setting up virtual environment...
echo
OUTPUT_FOLDER="./LONGGARCHFOR10"
OUTPUT_FOLDERVENV="./LONGGARCHFOR10"
set -e

pip3.8 install virtualenv # Assumes pip3 is installed!
python3.8 -m virtualenv $OUTPUT_FOLDERVENV/venv13
source $OUTPUT_FOLDERVENV/venv13/bin/activate


pip3.8 install arch
pip3.8 install yfinance
pip3.8 install joblib
pip3.8 install 'urllib3<2.0' # Downgrade urllib3 to a version before 2.0
pip3.8 install pysqlite3-binary # Install SQLite driver


# Run your Python script
python3.8 "$PYTHON_SCRIPT_PATH"

# Deactivate your virtual environment if you used one
deactivate
