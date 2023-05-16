import os
import zipfile
import requests
from tqdm import tqdm

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

# Define the URLs of the data to be downloaded
urls = [
    "http://images.cocodataset.org/zips/train2014.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
]

# Iterate over each URL
for url in urls:
    # Parse the filename
    filename = os.path.join(data_dir, url.split("/")[-1])
    
    # Download the file if it does not already exist
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        print(f"{filename} already exists, skipping download.")
        
    # Extract the zip file
    extract_zip(filename, data_dir)
