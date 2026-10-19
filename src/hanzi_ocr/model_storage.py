from pathlib import Path


def download_model(s3_client, bucket, prefix, dest_dir):
    """Downloads the model weights and encoders stored in S3 Bucket

    Parameters
    -----------
    s3_client: botocore.client.S3
        Initialized S3 client, created once with boto3.client("s3")
    bucket: str
        Name of the bucket where the model is stored
    prefix: str
        Folder of the model inside the bucket, ending in "/", e.g. "hanzi-ocr/v2/"
    dest_dir: str or Path
        Local folder where the files will be downloaded

    Returns
    -----------
    Path
        Local folder with the downloaded files, ready for load_model
    """
    dest_dir = Path(dest_dir)
    for name in ("checkpoint.pt", "le.json"):
        s3_client.download_file(bucket, f"{prefix}{name}", str(dest_dir / name))
    return dest_dir
