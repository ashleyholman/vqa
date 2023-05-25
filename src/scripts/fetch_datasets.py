import os
import zipfile
import requests
from tqdm import tqdm
import boto3
import socket
import subprocess
from botocore.exceptions import ClientError

import requests

# Define the URLs of the data to be downloaded
URLS = [
    # Training set
    "http://images.cocodataset.org/zips/train2014.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "s3://vqa-ap-southeast-1/datasets/train_embeddings.pt",

    # Validation set
    "http://images.cocodataset.org/zips/val2014.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "s3://vqa-ap-southeast-1/datasets/validation_embeddings.pt"
]

def is_ec2():
    return True
    """Check if we're on an EC2 instance"""
    try:
        # Make a request to the EC2 metadata service
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=0.02)
        # If the request succeeds without throwing an exception, we're on an EC2 instance
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # If the request throws an exception, we're likely not on an EC2 instance
        return False


def file_exists_in_s3(s3_url):
    """Check if file exists in S3"""
    s3 = boto3.resource('s3')
    bucket_name = s3_url.split('/')[2]
    s3_key = '/'.join(s3_url.split('/')[3:])
    bucket = s3.Bucket(bucket_name)

    objs = list(bucket.objects.filter(Prefix=s3_key))
    return len(objs) > 0 and objs[0].key == s3_key

def download_file_from_s3(s3_url, target_path):
    """Download a file from S3 using boto3"""
    s3 = boto3.resource('s3')
    bucket_name = s3_url.split('/')[2]
    s3_key = '/'.join(s3_url.split('/')[3:])

    try:
        print(f"Using s3 mirror for {s3_key}...")
        s3.Bucket(bucket_name).download_file(s3_key, target_path)
        return True
    except ClientError as e:
        print("The download failed: ", e)
        return False

def download_file(url, target_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def extract_zip(zip_file, destination):
    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"Unzipped {zip_file}")

# Define the data directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
print(f"Data dir: {data_dir}")

# Create the data directory if it does not exist
os.makedirs(data_dir, exist_ok=True)

# Check if we're on EC2
on_ec2 = is_ec2()

# Iterate over each URL
for url in URLS:
    # Parse the filename
    filename = os.path.join(data_dir, url.split("/")[-1])
    
    # Download the file if it does not already exist
    if not os.path.exists(filename):
        # If on EC2, try to download from the S3 mirror.
        # Not all files are on the mirror necessarily, so we'll fall back to the
        # original source if it's not there.
        s3_mirror_url = f"s3://vqa-ap-southeast-1/datasets/{filename.split('/')[-1]}"
        if on_ec2 and file_exists_in_s3(s3_mirror_url):
            download_file_from_s3(s3_mirror_url, filename)
            pass
        else:
            # If not on EC2 or if the file doesn't exist on S3, download from the original URL
            download_file(url, filename)
    else:
        print(f"{filename} already exists, skipping download.")
        
    # Extract the zip file
    extract_zip(filename, data_dir)

