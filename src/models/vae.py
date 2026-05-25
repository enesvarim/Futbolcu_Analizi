"""
src/models/vae.py
==================
FootballVAE: Variational Autoencoder + vae_loss
Model V2 için kullanılır.
"""
import torch
import torch.nn as nn

from src.config import VAE_LATENT_DIM, FEATURES


class FootballVAE(nn.Module):
    """
    Variational Autoencoder modeli.

    Encoder → (μ, log_var) → reparameterize → Decoder

    Avantajları:
    - Latent uzay sürekli ve düzgün dağılımlı → daha iyi kümeleme
    - β-annealing ile KL düzenleme kademeli artar
    - Inference'ta deterministik μ vektörü kullanılır
    """

    def __init__(self, input_dim: int = len(FEATURES), latent_dim: int = VAE_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim  = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.25),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
        )
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        z = μ + σ * ε, ε ~ N(0,1)
        Eğitimde stochastic, inference'ta deterministik (μ döner).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        recon   : torch.Tensor  [B, input_dim]
        mu      : torch.Tensor  [B, latent_dim]
        log_var : torch.Tensor  [B, latent_dim]
        """
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z       = self.reparameterize(mu, log_var)
        recon   = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministik μ vektörünü döner.
        Kümeleme ve benzerlik hesabı için kullanılır.
        """
        with torch.no_grad():
            h  = self.encoder(x)
            mu = self.fc_mu(h)
        return mu


# ---------------------------------------------------------------------------
# Loss Fonksiyonu
# ---------------------------------------------------------------------------
_huber = nn.HuberLoss(delta=1.0)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE toplam kaybı: Huber(rekon, girdi) + β * KL-Divergence

    KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))

    Returns
    -------
    total_loss, recon_loss, kl_loss
    """
    recon_loss = _huber(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Fabrika & Yükleme
# ---------------------------------------------------------------------------
def build_vae(
    input_dim: int  = len(FEATURES),
    latent_dim: int = VAE_LATENT_DIM,
) -> FootballVAE:
    """Fabrika fonksiyonu."""
    return FootballVAE(input_dim=input_dim, latent_dim=latent_dim)


def load_vae(
    path,
    input_dim: int  = len(FEATURES),
    latent_dim: int = VAE_LATENT_DIM,
    device: str     = "cpu",
) -> FootballVAE:
    """Kaydedilmiş VAE ağırlıklarını yükler."""
    model = build_vae(input_dim, latent_dim)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
