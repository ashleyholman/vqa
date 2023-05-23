#!/bin/bash
export AWS_DEFAULT_REGION=ap-southeast-1

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
echo "==> Make mini dataset"
./src/scripts/make_mini_dataset.sh
echo "==> Enter work loop"
nohup python src/scripts/do_work.py --num-dataloader-workers 8 >> train.log & 2>&1
