#!/bin/bash

echo "==> Setup python virtual environment"
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .

# Download data sets
echo
echo "==> Download data sets"
python src/scripts/fetch_datasets.py
