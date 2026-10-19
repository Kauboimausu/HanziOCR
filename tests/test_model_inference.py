import json

import cv2
import numpy as np
import pytest
import torch

from hanzi_ocr.hanzi_model import HanziModel
from hanzi_ocr.model_inference import (
    create_character_transformers,
    find_character_pairs,
    inference,
    load_model,
    preprocess_image,
)


@pytest.mark.parametrize("char", [b"notanimage", b""])
def test_preprocess_image_rejects_non_images(char):
    with pytest.raises(ValueError):
        preprocess_image(char)


@pytest.fixture(scope="module")
def converters():
    return create_character_transformers()


@pytest.mark.parametrize(
    "char, expected",
    [
        # Same in both scripts
        ("中", ("中", "中")),
        ("人", ("人", "人")),
        ("我", ("我", "我")),
        ("的", ("的", "的")),
        ("山", ("山", "山")),
        ("水", ("水", "水")),
        # Simplified input
        ("爱", ("爱", "愛")),
        ("国", ("国", "國")),
        ("书", ("书", "書")),
        ("学", ("学", "學")),
        ("门", ("门", "門")),
        ("马", ("马", "馬")),
        ("鱼", ("鱼", "魚")),
        ("龙", ("龙", "龍")),
        ("说", ("说", "說")),
        ("时", ("时", "時")),
        # Traditional input
        ("愛", ("爱", "愛")),
        ("國", ("国", "國")),
        ("書", ("书", "書")),
        ("學", ("学", "學")),
        ("門", ("门", "門")),
        ("馬", ("马", "馬")),
        ("魚", ("鱼", "魚")),
        ("龍", ("龙", "龍")),
        ("說", ("说", "說")),
        ("時", ("时", "時")),
    ],
)
def test_find_character_pairs_is_correct(char, expected, converters):
    assert find_character_pairs([char], *converters) == [expected]

def test_character_pairs_keep_input_order(converters):
    result = find_character_pairs(["爱", "中", "國"], *converters)
    assert result == [("爱", "愛"), ("中", "中"), ("国", "國")]

CODEPOINTS = [ord("中"), ord("文"), ord("字")]


@pytest.fixture 
def model_dir(tmp_path):
    model = HanziModel(len(CODEPOINTS))
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    (tmp_path / "le.json").write_text(json.dumps(CODEPOINTS))
    return tmp_path

def test_load_model_is_in_eval_mode(model_dir):
    model, _ = load_model(model_dir, "cpu")
    assert not model.training

def encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    assert ok 
    return buffer.tobytes()

def test_preprocess_image_returns_model_ready_batch():
    image = np.full((100, 80), 255, dtype=np.uint8)
    image[30:70, 20:60] = 0
    batch = preprocess_image(encode_png(image))
    assert batch.shape == (1, 1, 64, 64)
    assert batch.dtype == torch.float32

def test_preprocess_image_matches_training_normalization():
    image = np.full((100, 80), 255, dtype=np.uint8)
    image[30:70, 20:60] = 0
    batch = preprocess_image(encode_png(image))
    assert batch.max().item() == pytest.approx(1.0)
    assert batch.min().item() == pytest.approx(-1.0)

def test_load_model_accepts_str_path(model_dir):
    model, _ = load_model(str(model_dir), "cpu")
    assert not model.training


def test_load_model_returns_training_classes(model_dir):
    _, le = load_model(model_dir, "cpu")
    assert le.classes_.tolist() == CODEPOINTS


def test_load_model_fails_if_classes_do_not_match_checkpoint(model_dir):
    # le.json from a different model, with more classes than the checkpoint
    (model_dir / "le.json").write_text(json.dumps(CODEPOINTS + [ord("人")]))
    with pytest.raises(RuntimeError):
        load_model(model_dir, "cpu")


@pytest.mark.parametrize("missing_file", ["checkpoint.pt", "le.json"])
def test_load_model_fails_if_a_file_is_missing(model_dir, missing_file):
    (model_dir / missing_file).unlink()
    with pytest.raises(FileNotFoundError):
        load_model(model_dir, "cpu")


def test_preprocess_image_accepts_jpeg():
    image = np.full((100, 80), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    assert preprocess_image(buffer.tobytes()).shape == (1, 1, 64, 64)


def test_preprocess_image_converts_color_to_grayscale():
    image = np.full((100, 80, 3), 255, dtype=np.uint8)
    assert preprocess_image(encode_png(image)).shape == (1, 1, 64, 64)


class FixedLogitsModel(torch.nn.Module):
    """Stands in for the trained model, always returns the same logits"""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, X):
        return self.logits


@pytest.fixture
def fixed_le():
    """LabelEncoder for 中, 文, 爱, built the same way load_model builds it"""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.classes_ = np.array([ord("中"), ord("文"), ord("爱")])
    return le


@pytest.fixture
def image_batch():
    image = np.full((100, 80), 255, dtype=np.uint8)
    image[30:70, 20:60] = 0
    return preprocess_image(encode_png(image))


def test_inference_maps_highest_logit_to_its_character(fixed_le, image_batch, converters):
    # Highest logit at index 2, which is 爱 in fixed_le
    model = FixedLogitsModel([0.0, 1.0, 5.0])
    [(_, best_pair), *_] = inference(model, fixed_le, image_batch, 3, *converters)
    assert best_pair == ("爱", "愛")


def test_inference_returns_predictions_sorted_by_probability(fixed_le, image_batch, converters):
    model = FixedLogitsModel([1.0, 3.0, 2.0])
    results = inference(model, fixed_le, image_batch, 3, *converters)
    assert [pair for _, pair in results] == [("文", "文"), ("爱", "愛"), ("中", "中")]
    probs = [prob for prob, _ in results]
    assert probs == sorted(probs, reverse=True)


def test_inference_probabilities_are_python_floats(fixed_le, image_batch, converters):
    model = FixedLogitsModel([1.0, 3.0, 2.0])
    for prob, _ in inference(model, fixed_le, image_batch, 3, *converters):
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


def test_inference_full_top_k_probabilities_sum_to_one(fixed_le, image_batch, converters):
    model = FixedLogitsModel([1.0, 3.0, 2.0])
    results = inference(model, fixed_le, image_batch, 3, *converters)
    assert sum(prob for prob, _ in results) == pytest.approx(1.0)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_inference_returns_exactly_top_k_results(fixed_le, image_batch, converters, top_k):
    model = FixedLogitsModel([1.0, 3.0, 2.0])
    assert len(inference(model, fixed_le, image_batch, top_k, *converters)) == top_k


@pytest.mark.parametrize("top_k", [0, 4])
def test_inference_rejects_invalid_top_k(fixed_le, image_batch, converters, top_k):
    model = FixedLogitsModel([1.0, 3.0, 2.0])
    with pytest.raises(ValueError):
        inference(model, fixed_le, image_batch, top_k, *converters)


def test_inference_is_deterministic_with_real_model(model_dir, image_batch, converters):
    model, le = load_model(model_dir, "cpu")
    first = inference(model, le, image_batch, 3, *converters)
    second = inference(model, le, image_batch, 3, *converters)
    assert first == second


def test_inference_with_real_model_only_predicts_known_characters(
    model_dir, image_batch, converters
):
    model, le = load_model(model_dir, "cpu")
    known = {chr(codepoint) for codepoint in CODEPOINTS}
    for _, (simplified, _) in inference(model, le, image_batch, 3, *converters):
        assert simplified in known
