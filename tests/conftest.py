import json

import cv2
import numpy as np
import pytest
import torch

from hanzi_ocr.hanzi_model import HanziModel


@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def codepoints():
    return [ord("中"), ord("文"), ord("字")]

@pytest.fixture
def model_dir(tmp_path, codepoints):
    model = HanziModel(len(codepoints))
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    (tmp_path / "le.json").write_text(json.dumps(codepoints))
    return tmp_path


@pytest.fixture
def encode_png():
    """Factory fixture: returns a function that encodes an image as PNG bytes"""

    def encode(image):
        ok, buffer = cv2.imencode(".png", image)
        assert ok
        return buffer.tobytes()

    return encode