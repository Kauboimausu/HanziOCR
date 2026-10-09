import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, figure_save_location):
    n_epochs = len(history["train_losses"])

    plt.rc("font", size=14)
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("legend", fontsize=14)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    plt.plot(np.arange(n_epochs) + 0.5, history["train_losses"], label="Train Loss")
    plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], label="Train Metric")
    plt.plot(np.arange(n_epochs) + 0.5, history["valid_metrics"], label="Valid Metric")

    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.title("Training history")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(figure_save_location))


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
    verbose=1
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
            The metric that determines early stopping
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
    
    Returns
    history: dict
        - ``"train_losses"``: list of float
            Train losses, per epoch
        - ``"train_metrics"``: list of float
            Train metrics, per epoch
        - ``"valid_metrics"``: list of float
            Validation metrics, per epoch
    """
    

    best_valid_metric = 0.0
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

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_metric)

        if valid_metric >= best_valid_metric:
            epochs_without_improvement = 0
            best_valid_metric = valid_metric
            torch.save(model.state_dict(), checkpoint_path)
        elif epochs_without_improvement < patience:
            epochs_without_improvement += 1
        else:
            print(
                f"Out of patience, {epochs_without_improvement} without improvement, stopping training at epoch {epoch}"
            )
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history
