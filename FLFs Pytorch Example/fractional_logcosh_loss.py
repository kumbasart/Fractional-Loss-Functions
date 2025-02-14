import torch
import torch.nn as nn
from scipy.special import gamma
from util import inv_affine_sigmoid, affine_sigmoid


class FractionalLogCoshLoss(nn.Module):
    """
    Fractional Log-Cosh Loss Module.

    This module implements fractional log-cosh loss function.
    """

    def __init__(self,
                 alpha_init: float = 0.01, alpha_lo: float = 0.001, alpha_hi: float = 1,
                 c_init: float = 0.1, c_lo: float = 0.001, c_hi: float = 1, n_memory: int = 10, h: float = 0.1,
                 device: str = 'cpu'):
        super(FractionalLogCoshLoss, self).__init__()
        self.device = device
        self.n_memory = n_memory
        self.h = h
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.c_lo = c_lo
        self.c_hi = c_hi

        # Initialize latent parameters using the inverse affine sigmoid.
        latent_alpha_init = inv_affine_sigmoid(torch.tensor([alpha_init], device=device), alpha_lo, alpha_hi)
        self.latent_alpha = nn.Parameter(latent_alpha_init)
        latent_c_init = inv_affine_sigmoid(torch.tensor([c_init], device=device), c_lo, c_hi)
        self.latent_c = nn.Parameter(latent_c_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = affine_sigmoid(self.latent_alpha, self.alpha_lo, self.alpha_hi)
        c = affine_sigmoid(self.latent_c, self.c_lo, self.c_hi)
        return self.logcosh_loss(x - y, alpha, c, self.n_memory, self.h)

    def gamma_frac(self, z) -> torch.Tensor:
        """
        Compute the gamma function for input z.
        """
        return torch.tensor(gamma(z.detach().numpy()), dtype=torch.float32, device=self.device)

    def logcosh_loss(self, x: torch.Tensor, alpha: torch.Tensor, c: torch.Tensor, n_memory: int, h: float) -> torch.Tensor:

        sum_value = torch.zeros_like(x)
        for n in range(n_memory + 1):
            term = ((-1) ** n * self.gamma_frac(alpha + 1) *
                    torch.log(torch.cosh(torch.abs(x / c) - n * h))
                    ) / (self.gamma_frac(torch.tensor(n + 1, device=self.device)) * self.gamma_frac(1 - torch.tensor(n, device=self.device) + alpha))
            sum_value += term
        loss = sum_value / (h ** alpha)
        return loss
