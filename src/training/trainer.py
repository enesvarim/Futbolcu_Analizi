"""
src/training/trainer.py
========================
Ortak eğitim altyapısı — hem V1 (Autoencoder) hem V2 (VAE) için kullanılır.
Tek eğitim döngüsü, early stopping, checkpoint kaydetme.
"""
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import TRAINING

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoader yardımcısı
# ---------------------------------------------------------------------------
def make_loader(arr: np.ndarray, batch_size: int = TRAINING["batch_size"],
                shuffle: bool = True) -> DataLoader:
    """Numpy dizisinden PyTorch DataLoader oluşturur."""
    tensor = torch.FloatTensor(arr)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Autoencoder Trainer
# ---------------------------------------------------------------------------
def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Path,
    epochs: int   = TRAINING["epochs"],
    patience: int = TRAINING["patience"],
    lr: float     = TRAINING["lr"],
    weight_decay: float = TRAINING["weight_decay"],
    latent_l2: float    = 1e-4,
) -> dict:
    """
    Autoencoder (V1) eğitim döngüsü.

    Reconstruction Loss: MSE
    Latent Regularization: L2 (hafif küme sıkıştırma)
    Optimizer: AdamW + CosineAnnealingWarmRestarts

    Returns
    -------
    history : dict  {train_loss, val_loss, best_val_loss, test_mse}
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    train_loader = make_loader(X_train, shuffle=True)
    val_loader   = make_loader(X_val,   shuffle=False)
    test_loader  = make_loader(X_test,  shuffle=False)

    history = {"train_loss": [], "val_loss": [], "best_val_loss": None, "test_mse": None}
    best_val  = float("inf")
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_train = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            latent, out = model(batch)
            recon_loss  = criterion(out, batch)
            reg_loss    = latent_l2 * torch.mean(latent ** 2)
            (recon_loss + reg_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += recon_loss.item()

        avg_train = total_train / len(train_loader)

        # ── Validation ─────────────────────────────────────────
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                _, out = model(batch)
                total_val += criterion(out, batch).item()
        avg_val = total_val / len(val_loader)

        scheduler.step()
        history["train_loss"].append(round(avg_train, 6))
        history["val_loss"].append(round(avg_val, 6))

        if avg_val < best_val:
            best_val     = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            marker = "✓ BEST"
        else:
            patience_cnt += 1
            marker = ""

        if epoch % 10 == 0 or marker:
            log.info(
                f"Epoch [{epoch:3d}/{epochs}] | Train: {avg_train:.6f} | "
                f"Val: {avg_val:.6f} | Best: {best_val:.6f} | "
                f"Patience: {patience_cnt}/{patience} {marker}"
            )

        if patience_cnt >= patience:
            log.info(f"Early stopping — Epoch {epoch}")
            break

    # ── Test ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model.eval()
    total_test = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            _, out = model(batch)
            total_test += criterion(out, batch).item()
    avg_test = total_test / len(test_loader)

    history["best_val_loss"] = round(best_val, 6)
    history["test_mse"]      = round(avg_test, 6)
    elapsed = time.time() - t0
    log.info(f"Eğitim tamamlandı | Süre: {elapsed:.1f}s | Test MSE: {avg_test:.6f}")
    return history


# ---------------------------------------------------------------------------
# VAE Trainer
# ---------------------------------------------------------------------------
def train_vae(
    model: nn.Module,
    loss_fn: Callable,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Path,
    epochs: int       = TRAINING["epochs"],
    patience: int     = TRAINING["patience"],
    lr: float         = TRAINING["lr"],
    weight_decay: float = TRAINING["weight_decay"],
    beta_start: float = 0.0,
    beta_end: float   = 1.0,
    beta_warmup: int  = 50,
) -> dict:
    """
    VAE (V2) eğitim döngüsü.

    β-Annealing: beta, 0'dan 1'e kademeli artar (ilk beta_warmup epoch'ta)
    Loss: Huber (rekon) + β * KL-Divergence

    Returns
    -------
    history : dict
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    train_loader = make_loader(X_train, shuffle=True)
    val_loader   = make_loader(X_val,   shuffle=False)
    test_loader  = make_loader(X_test,  shuffle=False)

    history = {"train_loss": [], "val_loss": [], "best_val_loss": None, "test_loss": None}
    best_val     = float("inf")
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        beta = min(beta_end, beta_start + (beta_end - beta_start) * (epoch / beta_warmup))

        # ── Train ──────────────────────────────────────────────
        model.train()
        total_train = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, _, _ = loss_fn(recon, batch, mu, log_var, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        # ── Validation ─────────────────────────────────────────
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon, mu, log_var = model(batch)
                loss, _, _ = loss_fn(recon, batch, mu, log_var, beta)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)

        scheduler.step()
        history["train_loss"].append(round(avg_train, 6))
        history["val_loss"].append(round(avg_val, 6))

        if avg_val < best_val:
            best_val     = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            marker = "✓ BEST"
        else:
            patience_cnt += 1
            marker = ""

        if epoch % 10 == 0 or marker:
            log.info(
                f"Epoch [{epoch:3d}/{epochs}] β={beta:.3f} | "
                f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
                f"Best: {best_val:.5f} | Pat: {patience_cnt}/{patience} {marker}"
            )

        if patience_cnt >= patience:
            log.info(f"Early stopping — Epoch {epoch}")
            break

    # ── Test ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model.eval()
    total_test = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            recon, mu, log_var = model(batch)
            loss, _, _ = loss_fn(recon, batch, mu, log_var, beta=1.0)
            total_test += loss.item()
    avg_test = total_test / len(test_loader)

    history["best_val_loss"] = round(best_val, 6)
    history["test_loss"]     = round(avg_test, 6)
    elapsed = time.time() - t0
    log.info(f"VAE eğitimi tamamlandı | Süre: {elapsed:.1f}s | Test Loss: {avg_test:.6f}")
    return history
