"""
Dataset PyTorch para os batimentos segmentados do MIT-BIH,
split treino/val/teste e cálculo de pesos de classe para
lidar com o desbalanceamento.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ECGDataset(Dataset):
    """
    Wrapper simples: cada item é (janela, label).
    A janela é normalizada (z-score) por amostra.
    """

    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        window = self.X[idx]

        # z-score por batimento (cada janela normalizada individualmente)
        mean, std = window.mean(), window.std()
        if std > 0:
            window = (window - mean) / std

        # formato (seq_len, 1) -> RNN espera (seq_len, n_features)
        window = window.reshape(-1, 1)

        return torch.from_numpy(window), self.y[idx]


def split_data(X, y, val_size=0.15, test_size=0.15, seed=42):
    """
    Split estratificado (mantém a proporção de classes em cada conjunto).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=seed
    )

    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y, n_classes=5):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe,
    para usar na loss function (nn.CrossEntropyLoss(weight=...)).
    """
    counts = np.bincount(y, minlength=n_classes)
    total = counts.sum()

    # peso inversamente proporcional à frequência, normalizado
    weights = total / (n_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(X, y, batch_size=128, val_size=0.15, test_size=0.15, seed=42):
    """
    Função de conveniência: faz o split e retorna os 3 DataLoaders
    prontos + os pesos de classe.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, val_size=val_size, test_size=test_size, seed=seed
    )

    train_ds = ECGDataset(X_train, y_train)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(y_train)

    print(f'Treino: {len(train_ds)} | Validação: {len(val_ds)} | Teste: {len(test_ds)}')
    print(f'Pesos de classe (N,S,V,F,Q): {class_weights.tolist()}')

    return train_loader, val_loader, test_loader, class_weights