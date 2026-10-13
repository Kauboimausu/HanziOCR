import torch

from hanzi_ocr.HMNIST import HMNISTModel


def test_output_shape_is_batch_by_classes():
    model = HMNISTModel(n_classes=10)
    X = torch.randn(4, 1, 64, 64)
    assert model(X).shape == (4, 10)


def test_eval_mode_accepts_single_image():
    model = HMNISTModel(n_classes=10).eval()
    X = torch.randn(1, 1, 64, 64)
    assert model(X).shape == (1, 10)
