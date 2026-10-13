import pytest
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss, Identity, Linear
from torch.utils.data import DataLoader, TensorDataset

from hanzi_ocr.train_nn import evaluate, plot_training_history, train_with_early_stopping


class FakeMetric:
    """Stands in for a torchmetrics metric, returns a scripted validation value per epoch"""

    def __init__(self, valid_values):
        self.valid_values = list(valid_values)
        self.compute_calls = 0

    def reset(self):
        pass

    def update(self, preds, target):
        pass

    def compute(self):
        # compute is called twice per epoch: first for train, then for validation
        self.compute_calls += 1
        epoch = (self.compute_calls - 1) // 2
        return torch.tensor(self.valid_values[min(epoch, len(self.valid_values) - 1)])


@pytest.fixture
def loaders():
    X = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    return loader, loader


def run_training(loaders, tmp_path, valid_values, epochs, patience):
    train_loader, valid_loader = loaders
    model = Linear(4, 3)
    return train_with_early_stopping(
        "cpu",
        model,
        train_loader,
        valid_loader,
        CrossEntropyLoss(),
        FakeMetric(valid_values),
        torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
        checkpoint_path=tmp_path / "checkpoint.pt",
        epochs=epochs,
        patience=patience,
    )


def test_evaluate_perfect_predictions_score_one():
    y = torch.tensor([0, 1, 2, 1])
    X = torch.nn.functional.one_hot(y, num_classes=3).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    assert evaluate("cpu", Identity(), loader, metric) == pytest.approx(1.0)


def test_evaluate_puts_model_in_eval_mode():
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(torch.randn(2, 4), y), batch_size=2)
    model = Linear(4, 3)
    evaluate("cpu", model, loader, torchmetrics.Accuracy(task="multiclass", num_classes=3))
    assert not model.training


def test_improving_metric_runs_all_epochs(loaders, tmp_path):
    history = run_training(loaders, tmp_path, [0.1, 0.2, 0.3, 0.4, 0.5], epochs=5, patience=2)
    assert len(history["valid_metrics"]) == 5


def test_stops_after_patience_epochs_without_improvement(loaders, tmp_path):
    history = run_training(loaders, tmp_path, [0.5, 0.4, 0.4, 0.4, 0.4], epochs=10, patience=3)
    # 1 epoch that sets the best value + 3 epochs without improvement
    assert len(history["valid_metrics"]) == 4


def test_flat_metric_counts_as_no_improvement(loaders, tmp_path):
    history = run_training(loaders, tmp_path, [0.5], epochs=10, patience=3)
    assert len(history["valid_metrics"]) == 4


def test_checkpoint_is_saved_even_if_metric_is_zero(loaders, tmp_path):
    run_training(loaders, tmp_path, [0.0], epochs=2, patience=5)
    assert (tmp_path / "checkpoint.pt").exists()


def test_history_has_one_entry_per_epoch_in_each_list(loaders, tmp_path):
    history = run_training(loaders, tmp_path, [0.1, 0.2, 0.3], epochs=3, patience=2)
    assert {len(values) for values in history.values()} == {3}


def test_plot_training_history_saves_figure(tmp_path):
    history = {"train_losses": [2.0, 1.0], "train_metrics": [0.3, 0.6], "valid_metrics": [0.2, 0.5]}
    figure_path = tmp_path / "history.png"
    plot_training_history(history, figure_path)
    assert figure_path.exists()
