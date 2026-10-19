import boto3


def download_model(bucket, prefix, dest_dir):
    """Downloads the model weights and encoders stored in S3 Bucket"""
    s3 = boto3.client("s3")
    for name in ("checkpoint.pt", "le.json"):
        s3.download_file(bucket, f"{prefix}{name}", str(dest_dir / name))