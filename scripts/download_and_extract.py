import boto3
import zipfile
import os

# CONFIG 
BUCKET_NAME = "heydocai"         
OBJECT_KEY = "HeyDocAI_datasets/HeyDocAI_datasets.zip"     
LOCAL_ZIP = "HeyDocAI_datasets.zip"
EXTRACT_DIR = "datasets"

# DOWNLOAD FROM S3 
print(f"Downloading s3://{BUCKET_NAME}/{OBJECT_KEY} ...")
s3 = boto3.client("s3")
s3.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_ZIP)
print("Download complete.")

# EXTRACT ZIP
os.makedirs(EXTRACT_DIR, exist_ok=True)
print(f"Extracting to '{EXTRACT_DIR}/'...")
with zipfile.ZipFile(LOCAL_ZIP, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
print("Extraction complete.")
