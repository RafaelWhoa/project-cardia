"""
Loop de treino do classificador CNN+LSTM para arritmias.
Inclui validação por época, early stopping e checkpoint do melhor modelo.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def train_model(model, train_loader, val_loader, class_weights,
                 n_epochs=30, lr=1e-3, patience=5, device=None,
                 checkpoint_path='models/best_model.pt'):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando device: {device}')

    model = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_val_f1 = 0.0
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': []}

    for epoch in range(1, n_epochs + 1):
        # ---- treino ----
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- validação ----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1_macro'].append(val_f1)

        print(f'Época {epoch:02d} | train_loss: {train_loss:.4f} | '
              f'val_loss: {val_loss:.4f} | val_f1_macro: {val_f1:.4f}')

        # ---- checkpoint / early stopping ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  -> novo melhor modelo salvo (val_f1_macro: {val_f1:.4f})')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping na época {epoch} (sem melhora por {patience} épocas)')
                break

    print(f'\nMelhor val_f1_macro: {best_val_f1:.4f}')
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history


def evaluate_model(model, test_loader, device=None, class_names=('N', 'S', 'V', 'F', 'Q')):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    print('\n=== Relatório de classificação (conjunto de teste) ===')
    print(classification_report(all_labels, all_preds, target_names=list(class_names), digits=3))

    print('=== Matriz de confusão ===')
    print(confusion_matrix(all_labels, all_preds))

    return all_labels, all_preds