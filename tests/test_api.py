import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hanzi_ocr.api import app


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    with TestClient(app) as client:
        yield client


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_correctly_formatted_responses(client, encode_png):
    image = np.full((100, 80, 3), 128, dtype=np.uint8)
    image_bytes = encode_png(image)
    predictions = client.post(
        "/predict",
        files={"file": ("char.png", image_bytes, "image/png")},
        params={"top_k": 3},
    )
    assert predictions.status_code == 200
    predictions = predictions.json()
    assert len(predictions) == 3
    for prediction in predictions:
        assert "probability" in prediction
        assert isinstance(prediction["probability"], float)
        assert "simplified" in prediction
        assert isinstance(prediction["simplified"], str)
        assert "traditional" in prediction
        assert isinstance(prediction["traditional"], str)


def test_predict_returns_probabilities_sorted_in_decreasing_order(client, encode_png):
    image = np.full((100, 80, 1), 128, dtype=np.uint8)
    image_bytes = encode_png(image)
    predictions = client.post(
        "/predict",
        files={"file": ("char.png", image_bytes, "image/png")},
        params={"top_k": 3},
    )
    assert predictions.status_code == 200
    predictions = predictions.json()
    probabilities = [prediction["probability"] for prediction in predictions]
    assert sorted(probabilities, reverse=True) == probabilities


def test_predict_returns_error_code_with_range_invalid_top_k_values(client, encode_png):
    image = np.full((100, 80, 1), 128, dtype=np.uint8)
    image_bytes = encode_png(image)
    response = client.post(
        "/predict",
        files={"file": ("char.png", image_bytes, "image/png")},
        params={"top_k": 100},
    )
    assert response.status_code == 400


def test_predict_returns_error_code_with_type_invalid_top_k_values(client, encode_png):
    image = np.full((100, 80, 1), 128, dtype=np.uint8)
    image_bytes = encode_png(image)
    response = client.post(
        "/predict",
        files={"file": ("char.png", image_bytes, "image/png")},
        params={"top_k": "abc"},
    )
    assert response.status_code == 422


@pytest.fixture
def png_bytes(encode_png):
    image = np.full((100, 80), 255, dtype=np.uint8)
    image[30:70, 20:60] = 0
    return encode_png(image)


@pytest.mark.parametrize("top_k", [0, -1])
def test_predict_rejects_top_k_below_one(client, png_bytes, top_k):
    response = client.post(
        "/predict",
        files={"file": ("char.png", png_bytes, "image/png")},
        params={"top_k": top_k},
    )
    assert response.status_code == 400


def test_predict_with_top_k_equal_to_classes_returns_every_class(
    client, png_bytes, codepoints
):
    response = client.post(
        "/predict",
        files={"file": ("char.png", png_bytes, "image/png")},
        params={"top_k": len(codepoints)},
    )
    assert response.status_code == 200
    predicted = {prediction["simplified"] for prediction in response.json()}
    assert predicted == {chr(codepoint) for codepoint in codepoints}


@pytest.mark.parametrize("content", [b"notanimage", b""])
def test_predict_rejects_files_that_are_not_images(client, content):
    response = client.post(
        "/predict",
        files={"file": ("char.png", content, "image/png")},
        params={"top_k": 3},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_without_file_returns_validation_error(client):
    response = client.post("/predict", params={"top_k": 3})
    assert response.status_code == 422


class CopyingS3Client:
    """Fake S3 client that serves the files of a local model folder"""

    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.requested = []

    def download_file(self, bucket, key, filename):
        self.requested.append((bucket, key))
        shutil.copy(self.source_dir / Path(key).name, filename)


@pytest.fixture
def fake_s3(model_dir, tmp_path, monkeypatch):
    """S3 mode: no MODEL_DIR, a bucket set, and boto3 replaced by a fake client"""
    fake_client = CopyingS3Client(model_dir)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.setenv("MODEL_BUCKET", "my-bucket")
    monkeypatch.setattr("hanzi_ocr.api.boto3.client", lambda service: fake_client)

    # Downloads go inside tmp_path instead of the system's temp folder
    download_dir = tmp_path / "downloaded"
    download_dir.mkdir()
    monkeypatch.setattr("hanzi_ocr.api.mkdtemp", lambda: str(download_dir))
    return fake_client


def test_startup_downloads_model_from_default_s3_location(fake_s3):
    with TestClient(app):
        pass
    assert sorted(fake_s3.requested) == [
        ("my-bucket", "hanziOCR/V1/checkpoint.pt"),
        ("my-bucket", "hanziOCR/V1/le.json"),
    ]


def test_startup_strips_slashes_from_model_name_and_version(fake_s3, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "/my_model/")
    monkeypatch.setenv("MODEL_VERSION", "V2/")
    with TestClient(app):
        pass
    assert sorted(fake_s3.requested) == [
        ("my-bucket", "my_model/V2/checkpoint.pt"),
        ("my-bucket", "my_model/V2/le.json"),
    ]


def test_predict_works_with_model_downloaded_from_s3(fake_s3, png_bytes):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("char.png", png_bytes, "image/png")},
            params={"top_k": 3},
        )
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_startup_fails_without_model_dir_or_bucket(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    with pytest.raises(KeyError), TestClient(app):
        pass
