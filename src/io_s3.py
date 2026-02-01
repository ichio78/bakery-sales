import os
import boto3

s3 = boto3.client("s3")

def parse_s3_uri(s3_uri: str):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri.replace("s3://", "", 1)
    bucket, key = path.split("/", 1)
    return bucket, key

def download_s3_to_local(s3_uri: str, local_path: str):
    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def upload_local_to_s3(local_path: str, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    s3.upload_file(local_path, bucket, key)
