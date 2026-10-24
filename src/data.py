import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_loaders(batch_size: int, seed: int):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # normalize to range 0..1

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    def make(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return make(X_train, y_train, True), make(X_val, y_val, False)
