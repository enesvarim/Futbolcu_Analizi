"""
src/training/trainer.py
========================
Ortak egitim altyapisi — hem V1 (Autoencoder) hem V2 (VAE) icin kullanilir.
Tek egitim dongusu, early stopping, checkpoint kaydetme.
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
# DataLoader yardimcisi
# ---------------------------------------------------------------------------
def make_loader(arr: np.ndarray, batch_size: int = TRAINING["batch_size"],
                shuffle: bool = True) -> DataLoader:
    """Numpy dizisinden PyTorch DataLoader olusturur."""
    tensor = torch.FloatTensor(arr)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Autoencoder Trainer (V1)
# ---------------------------------------------------------------------------
def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Path,
    epochs: int         = TRAINING["epochs"],
    patience: int       = TRAINING["patience"],
    lr: float           = TRAINING["lr"],
    weight_decay: float = TRAINING["weight_decay"],
    latent_l2: float    = 1e-4,
) -> dict:
    """
    Autoencoder (V1) egitim dongusu.

    Reconstruction Loss: MSE
    Latent Regularization: L2 (hafif kume sikistirma)
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
    best_val     = float("inf")
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # -- Train ----------------------------------------------------------
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

        # -- Validation -----------------------------------------------------
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
            marker = "BEST"
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
            log.info(f"Early stopping - Epoch {epoch}")
            break

    # -- Test ---------------------------------------------------------------
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
    log.info(f"Egitim tamamlandi | Sure: {elapsed:.1f}s | Test MSE: {avg_test:.6f}")
    return history


# ---------------------------------------------------------------------------
# VAE Trainer (V2)
# ---------------------------------------------------------------------------
def train_vae(
    model: nn.Module,
    loss_fn: Callable,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Path,
    epochs: int         = TRAINING["epochs"],
    patience: int       = TRAINING["patience"],
    lr: float           = TRAINING["lr"],
    weight_decay: float = TRAINING["weight_decay"],
    beta_start: float   = 0.0,
    beta_end: float     = 1.0,
    beta_warmup: int    = 50,
) -> dict:
    """
    VAE (V2) egitim dongusu.

    beta-Annealing: beta, 0'dan beta_end'e kademeli artar (ilk beta_warmup epoch'ta).
    Loss (train): Huber (rekon) + beta * KL-Divergence
    Loss (val)  : Sadece Huber (rekon) — beta'dan bagimsiz, kararli sinyal.

    Iyilestirmeler:
    - Validation sadece rekonstruksiyon kaybi olcuyor (beta=0) → kararli early stopping
    - Warmup bitmeden patience sayilmiyor → beta annealing sirasinda erken dur engellendi
    - Test loss hem rekon (val ile kiyaslanabilir) hem tam loss (beta_end ile) olarak raporlanir

    Returns
    -------
    history : dict  {train_loss, val_loss, recon_loss, kl_loss,
                     best_val_loss, test_loss, test_loss_full}
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    train_loader = make_loader(X_train, shuffle=True)
    val_loader   = make_loader(X_val,   shuffle=False)
    test_loader  = make_loader(X_test,  shuffle=False)

    history = {
        "train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": [],
        "best_val_loss": None, "test_loss": None, "test_loss_full": None,
    }
    best_val     = float("inf")
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        beta = min(beta_end, beta_start + (beta_end - beta_start) * (epoch / beta_warmup))

        # -- Train ----------------------------------------------------------
        model.train()
        total_train, total_recon, total_kl = 0.0, 0.0, 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, rl, kl = loss_fn(recon, batch, mu, log_var, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()
            total_recon += rl.item()
            total_kl    += kl.item()
        avg_train = total_train / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl    = total_kl    / len(train_loader)

        # -- Validation (sadece rekon kaybi — beta'dan bagimsiz sinyal) -----
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon, mu, log_var = model(batch)
                _, rl, _ = loss_fn(recon, batch, mu, log_var, beta=0.0)
                total_val += rl.item()
        avg_val = total_val / len(val_loader)

        scheduler.step()
        history["train_loss"].append(round(avg_train, 6))
        history["val_loss"].append(round(avg_val, 6))
        history["recon_loss"].append(round(avg_recon, 6))
        history["kl_loss"].append(round(avg_kl, 6))

        if avg_val < best_val:
            best_val     = avg_val
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            marker = "BEST"
        else:
            # Warmup bitmeden patience saymaya baslama — beta annealing stabil degil
            if epoch > beta_warmup:
                patience_cnt += 1
            marker = ""

        if epoch % 10 == 0 or marker:
            log.info(
                f"Epoch [{epoch:3d}/{epochs}] beta={beta:.3f} | "
                f"Train: {avg_train:.5f} | Val(Rekon): {avg_val:.5f} | "
                f"Best: {best_val:.5f} | Pat: {patience_cnt}/{patience} {marker}"
            )

        if patience_cnt >= patience:
            log.info(f"Early stopping - Epoch {epoch}")
            break

    # -- Test ---------------------------------------------------------------
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model.eval()
    total_test_recon, total_test_full = 0.0, 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            recon, mu, log_var = model(batch)
            # Rekon bazli test (val ile kiyaslanabilir)
            _, rl, _ = loss_fn(recon, batch, mu, log_var, beta=0.0)
            total_test_recon += rl.item()
            # Tam loss (beta_end ile)
            full, _, _ = loss_fn(recon, batch, mu, log_var, beta=beta_end)
            total_test_full += full.item()
    avg_test_recon = total_test_recon / len(test_loader)
    avg_test_full  = total_test_full  / len(test_loader)

    history["best_val_loss"]  = round(best_val, 6)
    history["test_loss"]      = round(avg_test_recon, 6)
    history["test_loss_full"] = round(avg_test_full, 6)
    elapsed = time.time() - t0
    log.info(
        f"VAE egitimi tamamlandi | Sure: {elapsed:.1f}s "
        f"| Test Rekon Loss: {avg_test_recon:.6f} | Test Full Loss: {avg_test_full:.6f}"
    )
    return history
