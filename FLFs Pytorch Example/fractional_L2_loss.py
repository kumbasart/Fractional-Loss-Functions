import torch
import torch.nn as nn
from scipy.special import gamma
from util import inv_affine_sigmoid, affine_sigmoid


class FractionalL2Loss(nn.Module):
    """
    Fractional L2 Loss Module.

    This module implements the fractional L2 loss function.
    The loss is computed as:

        L(x) = (|x/c|^(2 - α)) / Γ(3 - α) * c²

    where:
      - α is the fractional order parameter.
      - c is the scale parameter.

    Both parameters are constrained within specified bounds using an affine sigmoid transformation as explained in the
    paper.
    """

    def __init__(self,
                 alpha_init: float = 0.01, alpha_lo: float = 0.001, alpha_hi: float = 1,
                 c_init: float = 0.1, c_lo: float = 0.001, c_hi: float = 1,
                 device: str = 'cpu'):
        super(FractionalL2Loss, self).__init__()
        self.device = device
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
        # Transform latent parameters to constrained values.
        alpha = affine_sigmoid(self.latent_alpha, self.alpha_lo, self.alpha_hi)
        c = affine_sigmoid(self.latent_c, self.c_lo, self.c_hi)
        return self.l2_loss(x - y, alpha, c)

    def gamma_frac(self, z) -> torch.Tensor:
        """
        Compute the gamma function for input z.
        """
        return torch.tensor(gamma(z.detach().numpy()), dtype=torch.float32, device=self.device)

    def l2_loss(self, x: torch.Tensor, alpha: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        loss_val = (torch.abs(x / c).pow(2 - alpha) / self.gamma_frac(3 - alpha)) * (c ** 2)
        return loss_val
