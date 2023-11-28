import torch

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=10):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        """End time of the SDE."""
        return 1

    def sde(self, x, t):
        """Define the stochastic differential equation

        Args:
            x: mini-batch of images
            t: time

        Returns:
            the drift(f) and the diffusion coefficient (G) 

        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$

        Args:
            x: mini-batch of images
            t: randomly sampled time

        Returns:
            mean and the standard deviation of the marginal probability distribution.

        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$.

        Args:
            shape: shape of the mini-batch

        Returns:
            Pure Gaussian noise of shape of the mini-batch
        """
        return torch.randn(*shape)