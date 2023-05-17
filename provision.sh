#!/bin/bash

echo "==> Setup python virtual environment"
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
# Downgrade urllib3 to avoid SSL version error
pip install 'urllib3<2.0'

# Download data sets
echo
echo "==> Download data sets"
python src/scripts/fetch_datasets.py
