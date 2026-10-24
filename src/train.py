import random
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data import get_loaders
from src.model import MLP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    train_loader, val_loader = get_loaders(cfg["batch_size"], cfg["seed"])

    model = MLP(hidden=cfg["hidden_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment("digits")
    with mlflow.start_run():
        mlflow.log_params(cfg)

        for epoch in range(cfg["epochs"]):
            model.train()
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                optimizer.step()

            acc = evaluate(model, val_loader)
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            print(f"epoch {epoch + 1}: val_acc={acc:.4f}")

        torch.save(model.state_dict(), cfg["model_path"])
        mlflow.log_artifact(cfg["model_path"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
