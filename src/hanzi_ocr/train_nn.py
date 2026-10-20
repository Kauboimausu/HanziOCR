import math

import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

# Values early stopping can monitor: the history list it reads and whether higher or lower is better
MONITORS = {
    "valid_metric": ("valid_metrics", "max"),
    "train_loss": ("train_losses", "min"),
}

# Colors for the training history plot
TRAIN_COLOR = "#2a78d6"
VALID_COLOR = "#eb6834"
REFERENCE_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"


def find_best_epoch(history, monitor):
    """Returns the epoch, counting from 1, with the best value of the monitored quantity

    Parameters
    -----------
    history: dict
        History returned by train_with_early_stopping
    monitor: str
        One of the keys of MONITORS

    Returns
    -----------
    int
        Best epoch, the first one if several are tied
    """
    history_key, mode = MONITORS[monitor]
    values = history[history_key]
    best_value = max(values) if mode == "max" else min(values)
    return values.index(best_value) + 1


def plot_training_history(
    history, figure_save_location, metric_name="metric", monitor="valid_metric", num_classes=None
):
    """Saves a figure with the loss on the left and the metric on the right

    Parameters
    -----------
    history: dict
        History returned by train_with_early_stopping
    figure_save_location: str or Path
        Path of the image file to be saved, e.g. a .png
    metric_name: str
        Name of the metric, used in the labels
    monitor: str
        Quantity early stopping monitored, used to mark the epoch of the saved checkpoint
    num_classes: int or None
        If given, the loss of a model guessing at random, ln(num_classes), is drawn as a reference
    """
    epochs = range(1, len(history["train_losses"]) + 1)
    best_epoch = find_best_epoch(history, monitor)

    # Figure instead of pyplot: no global state, no display needed, safe to call many times
    fig = Figure(figsize=(12, 4.5), layout="constrained", facecolor=SURFACE_COLOR)
    loss_ax, metric_ax = fig.subplots(1, 2)

    loss_ax.plot(epochs, history["train_losses"], color=TRAIN_COLOR, linewidth=2, marker="o", markersize=5, label="Train loss")
    if num_classes is not None:
        loss_ax.axhline(math.log(num_classes), color=REFERENCE_COLOR, linestyle=":", linewidth=1.5, label="Random guessing")
    loss_ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

    metric_ax.plot(epochs, history["train_metrics"], color=TRAIN_COLOR, linewidth=2, marker="o", markersize=5, label=f"Train {metric_name}")
    metric_ax.plot(epochs, history["valid_metrics"], color=VALID_COLOR, linewidth=2, marker="o", markersize=5, label=f"Validation {metric_name}")
    metric_ax.set(title=metric_name.capitalize(), xlabel="Epoch", ylabel=metric_name)
    metric_ax.set_ylim(bottom=0)

    for ax in (loss_ax, metric_ax):
        ax.axvline(best_epoch, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label=f"Saved checkpoint (epoch {best_epoch})")
        ax.set_facecolor(SURFACE_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False)

    fig.suptitle(f"Training history, early stopping on {monitor.replace('_', ' ')}")
    fig.savefig(figure_save_location, dpi=120)


def evaluate(device, model, test_batches, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in test_batches:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            metric.update(preds, y_batch)

    return metric.compute().item()


def train_with_early_stopping(
    device,
    model,
    train_loader,
    valid_loader,
    criterion,
    metric,
    optimizer,
    scheduler,
    checkpoint_path,
    epochs=100,
    patience=10,
    monitor="valid_metric",
):
    """Trains the model with early stopping and saves the checkpoint

    Parameters
    -----------
        device: str
            The device used for training
        model: torch.nn.Module
            The model
        train_loader:
            All ready training loader
        valid_loader:
            All ready validation loader
        criterion:
            The loss function, guides training
        metric:
            The metric computed on the training and validation sets
        optimizer:
            The optimizer to be used to speed up training, if any
        scheduler:
            The scheduler to be used to optimize training, if any
        checkpoint_path: str
            Where the checkpoints, weights for the model, will be saved
        epochs: int
            Maximum amount of epochs the model can be trained for
        patience: int
            Maximum patience of epochs without improvement for early stopping
        monitor: str
            What early stopping and the checkpoint follow, one of:
                ``"valid_metric"``: the validation metric, higher is better
                ``"train_loss"``: the training loss, lower is better

    Returns
    -----------
    history: dict
        - ``"train_losses"``: list of float
            Train losses, per epoch
        - ``"train_metrics"``: list of float
            Train metrics, per epoch
        - ``"valid_metrics"``: list of float
            Validation metrics, per epoch

    Raises
    -----------
    ValueError
        If monitor is not one of the accepted values
    """
    if monitor not in MONITORS:
        raise ValueError(f"monitor must be one of {list(MONITORS)}, got {monitor!r}")
    history_key, mode = MONITORS[monitor]

    best_value = -math.inf if mode == "max" else math.inf
    epochs_without_improvement = 0
    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": [],
    }

    for epoch in range(epochs):
        metric.reset()
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")

        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(preds, y_batch)

            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

        total_train_loss = total_loss / len(train_loader)
        print(f"Train loss of {total_train_loss} for epoch {epoch+1}")
        train_metric = metric.compute().item()
        valid_metric = evaluate(device, model, valid_loader, metric)
        print(f"Validation metric of {valid_metric} for epoch {epoch+1}")

        history["train_losses"].append(total_train_loss)
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(valid_metric)

        current_value = history[history_key][-1]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(current_value)

        improved = current_value > best_value if mode == "max" else current_value < best_value
        if improved:
            epochs_without_improvement = 0
            best_value = current_value
            torch.save(model.state_dict(), checkpoint_path)
        elif epochs_without_improvement < (patience-1):
            epochs_without_improvement += 1
        else:
            print(
                f"Out of patience, {epochs_without_improvement} without improvement, stopping training at epoch {epoch+1}"
            )
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history
