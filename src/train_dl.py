"""
train_dl.py
Réseau de neurones (MLP) avec PyTorch pour la prédiction de maladies cardiaques.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC_PATH = ROOT / "data" / "processed"
DL_PATH   = ROOT / "models" / "dl"
DL_PATH.mkdir(parents=True, exist_ok=True)

# ── Hyperparamètres ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-3
HIDDEN_DIMS = [128, 64, 32]
DROPOUT     = 0.3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Architecture ─────────────────────────────────────────────────────────────
class HeartMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Utilitaires ──────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(PROC_PATH / "X_train.npy").astype(np.float32)
    X_test  = np.load(PROC_PATH / "X_test.npy").astype(np.float32)
    y_train = np.load(PROC_PATH / "y_train.npy").astype(np.float32)
    y_test  = np.load(PROC_PATH / "y_test.npy").astype(np.float32)
    return X_train, X_test, y_train, y_test


def make_loaders(X_train, y_train, X_test, y_test):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
    return train_dl, test_dl


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_proba, all_labels = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        proba = model(xb).cpu().numpy()
        preds = (proba >= 0.5).astype(int)
        all_proba.extend(proba)
        all_preds.extend(preds)
        all_labels.extend(yb.numpy().astype(int))
    return np.array(all_labels), np.array(all_preds), np.array(all_proba)


def main():
    print(f"Device : {DEVICE}\n")
    X_train, X_test, y_train, y_test = load_data()
    train_dl, test_dl = make_loaders(X_train, y_train, X_test, y_test)

    input_dim = X_train.shape[1]
    model     = HeartMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCELoss()

    print(f"Modèle :\n{model}\n")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_dl, optimizer, criterion)
        labels, preds, proba = evaluate(model, test_dl)
        acc = accuracy_score(labels, preds)
        scheduler.step(loss)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), DL_PATH / "best_mlp.pt")

        if epoch % 10 == 0:
            auc = roc_auc_score(labels, proba)
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  acc={acc:.4f}  auc={auc:.4f}")

    # Évaluation finale avec le meilleur modèle
    print(f"\n{'='*50}")
    print("  ÉVALUATION FINALE (meilleur modèle)")
    print(f"{'='*50}")
    model.load_state_dict(torch.load(DL_PATH / "best_mlp.pt", map_location=DEVICE))
    labels, preds, proba = evaluate(model, test_dl)
    print(f"  Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(labels, proba):.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['Sain', 'Malade'])}")
    print(f"\n  Modèle sauvegardé : {DL_PATH / 'best_mlp.pt'}")


if __name__ == "__main__":
    main()
