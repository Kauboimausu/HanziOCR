from pathlib import Path

import pytest

from hanzi_ocr.model_storage import download_model


class FakeS3Client:
    def __init__(self):
        self.requested = []

    def download_file(self, bucket, key, filename):
        self.requested.append((bucket, key))
        Path(filename).write_text("fake")


@pytest.fixture
def initialize_storage(tmp_path):
    fake_client = FakeS3Client()
    dest_dir = download_model(fake_client, "my-bucket", "hanzi-ocr/v2/", tmp_path)
    return dest_dir, fake_client


def test_download_model_returns_folder(initialize_storage, tmp_path):
    dest_dir, _ = initialize_storage
    assert dest_dir == tmp_path


def test_download_model_downloads_files(initialize_storage):
    dest_dir, _ = initialize_storage
    assert sorted(p.name for p in dest_dir.iterdir()) == ["checkpoint.pt", "le.json"]


def test_download_model_downloads_retrieves_keys(initialize_storage):
    _, fake_s3 = initialize_storage
    assert ("my-bucket", "hanzi-ocr/v2/checkpoint.pt") in fake_s3.requested
    assert ("my-bucket", "hanzi-ocr/v2/le.json") in fake_s3.requested
