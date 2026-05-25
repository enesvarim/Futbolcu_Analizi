"""
src/models/autoencoder.py
==========================
FootballAutoencoder: Encoder → 12-D latent uzay → Decoder
Model V1 için kullanılır.
"""
import torch
import torch.nn as nn

from src.config import AE_LATENT_DIM, FEATURES


class FootballAutoencoder(nn.Module):
    """
    Derin Autoencoder modeli.

    Encoder : input_dim → 256 → 128 → 64 → 32 → latent_dim
    Decoder : latent_dim → 32 → 64 → 128 → 256 → input_dim

    BatchNorm + LeakyReLU + Dropout kombinasyonu overfitting'i önler
    ve latent uzayın kümeleme için daha ayırt edici olmasını sağlar.
    """

    def __init__(self, input_dim: int = len(FEATURES), latent_dim: int = AE_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim  = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.LeakyReLU(0.1),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),  nn.LeakyReLU(0.1),
            nn.Linear(32, 64),          nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        latent : torch.Tensor  [B, latent_dim]
        recon  : torch.Tensor  [B, input_dim]
        """
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return latent, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece encoder çıktısını döner (inference/benzerlik için)."""
        return self.encoder(x)


def build_autoencoder(
    input_dim: int = len(FEATURES),
    latent_dim: int = AE_LATENT_DIM,
) -> FootballAutoencoder:
    """Fabrika fonksiyonu — model oluşturur."""
    return FootballAutoencoder(input_dim=input_dim, latent_dim=latent_dim)


def load_autoencoder(
    path,
    input_dim: int = len(FEATURES),
    latent_dim: int = AE_LATENT_DIM,
    device: str = "cpu",
) -> FootballAutoencoder:
    """
    Kaydedilmiş ağırlıkları yükler ve eval moduna geçirir.

    Parameters
    ----------
    path : str | Path
        .pth dosyasının yolu.
    """
    model = build_autoencoder(input_dim, latent_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
