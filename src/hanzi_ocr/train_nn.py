import torch
import os

def evaluate(device, model, test_batches, metric):
    metric.reset
    for X_batch, y_batch in test_batches:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        metric.update(preds, y_batch)

        return metric.compute().item()

def train_with_early_stopping(device, model, train_loader, valid_loader, criterion, metric, optimizer, scheduler, checkpoint_folder, epochs=100, patience=10):
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pt")

    best_valid_metric = 0.0
    epochs_without_improvement = 0
    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": [],
    }

    for _ in range(epochs):
        metric.reset()
        model.train()

        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(preds, y_batch)

            total_train_loss = total_loss / len(train_loader)
            train_metric = metric.compute().item()
            valid_metric = evaluate(device, model, valid_loader, metric)


            history["train_losses"].append(total_train_loss)
            history["train_metrics"].append(train_metric)
            history["valid_metrics"].append(valid_metric)

            scheduler.step()

            if valid_metric >= best_valid_metric:
                epochs_without_improvement = 0
                best_valid_metric = valid_metric
                torch.save(model.state_dict(), checkpoint_path)
            elif epochs_without_improvement < patience:
                patience += 1
            else:
                print("Out of patience, stopping training")
                break

            model.load_state_dict(torch.load(checkpoint_path))
            return history
