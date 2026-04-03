"""
noise_scheduler.py — Linear DDPM noise scheduler with v-prediction support.

Implements:
  • Standard DDPM forward process (add_noise).
  • V-prediction target computation (get_v_target).
  • X0 / noise recovery from v prediction (predict_x0_and_noise_from_v).
  • DDPM reverse sampling with v-prediction (sample_prev_timestep_v).
  • DDIM reverse sampling with v-prediction (sample_prev_timestep_v_ddim).
  • SNR-based loss weighting (get_snr_weight).
  • I2SB interpolation coefficients for bridge-based diffusion (q_sample).
"""

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def unsqueeze_xdim(tensor, xdim):
    """Unsqueezes `tensor` once per dimension in `xdim` to enable broadcasting."""
    for _ in range(len(xdim)):
        tensor = tensor.unsqueeze(-1)
    return tensor


class LinearNoiseScheduler:
    """
    Linear beta-schedule noise scheduler with v-prediction and I2SB support.

    The beta schedule runs linearly from beta_start to beta_end over num_timesteps
    steps, following the original DDPM formulation.
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start    = beta_start
        self.beta_end      = beta_end

        self.betas                        = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas                       = 1. - self.betas
        self.alpha_cum_prod               = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cum_prod          = torch.sqrt(self.alpha_cum_prod).to(device)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod).to(device)
        self._compute_i2sb_coefficients()

    def _compute_i2sb_coefficients(self):
        """
        Pre-computes the I2SB interpolation coefficients:
            mu_x0[t]  — weight on the clean image x0 at timestep t.
            mu_x1[t]  — weight on the noisy boundary x1 at timestep t.
            std_sb[t]  — noise standard deviation for the stochastic bridge.

        Coefficients are derived from alpha_cum_prod so that boundary conditions
        hold: at t=0 the state is purely x0; at t=T it is purely x1.
        """
        alpha_t = self.alpha_cum_prod
        self.mu_x0 = torch.sqrt(alpha_t)        # weight on x0: high at t=0, low at t=T
        self.mu_x1 = torch.sqrt(1 - alpha_t)    # weight on x1: low at t=0, high at t=T

        # Bridge noise: peaks at intermediate timesteps, zero at both ends.
        t = torch.linspace(0, 1, self.num_timesteps).to(device)
        self.std_sb = torch.sqrt(t * (1 - t)).to(device) * 0.5

        # Enforce boundary conditions.
        self.mu_x0[0]  = 1.0;  self.mu_x1[0]  = 0.0;  self.std_sb[0]  = 0.0
        self.mu_x0[-1] = 0.0;  self.mu_x1[-1] = 1.0;  self.std_sb[-1] = 0.0

    def q_sample(self, step, x0, x1, ot_ode=False):
        """
        I2SB forward process: samples q(x_t | x0, x1).

        Interpolates between the source distribution x0 and the target x1,
        optionally adding bridge noise.

        Args:
            step   (int | torch.Tensor): Timestep(s).
            x0     (torch.Tensor): Source (damaged) image.
            x1     (torch.Tensor): Target (clean) image.
            ot_ode (bool): If True, uses the deterministic OT-ODE path (no noise).

        Returns:
            torch.Tensor: Intermediate state x_t.
        """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        if isinstance(step, int):
            step = torch.tensor([step] * batch, device=x0.device)
        elif step.dim() == 0:
            step = step.repeat(batch)

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)

        return xt.detach()

    def add_noise(self, original, noise, t):
        """
        DDPM forward process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            original (torch.Tensor): Clean image of shape (B, C, H, W).
            noise    (torch.Tensor): Gaussian noise of the same shape.
            t        (torch.Tensor): Timestep indices of shape (B,).

        Returns:
            torch.Tensor: Noisy image x_t.
        """
        original_shape = original.shape
        batch_size     = original_shape[0]

        sqrt_alpha = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_m = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_m = sqrt_one_m.unsqueeze(-1)

        return sqrt_alpha * original + sqrt_one_m * noise

    def get_v_target(self, original, noise, t):
        """
        Computes the v-prediction target.

        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0

        Args:
            original (torch.Tensor): Clean image x_0, shape (B, C, H, W).
            noise    (torch.Tensor): Gaussian noise epsilon, same shape.
            t        (torch.Tensor): Timestep indices, shape (B,).

        Returns:
            torch.Tensor: v target of the same shape as original.
        """
        original_shape = original.shape
        batch_size     = original_shape[0]

        sqrt_alpha = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_m = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_m = sqrt_one_m.unsqueeze(-1)

        return sqrt_alpha * noise - sqrt_one_m * original

    def predict_x0_and_noise_from_v(self, xt, v_pred, t):
        """
        Recovers the clean image x_0 and noise epsilon from a v prediction.

        x_0     = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
        epsilon = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v

        Args:
            xt     (torch.Tensor): Noisy image at timestep t, shape (B, C, H, W).
            v_pred (torch.Tensor): Model's v prediction, same shape.
            t      (torch.Tensor): Timestep indices, shape (B,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (x0_pred, noise_pred), both clamped.
        """
        original_shape = xt.shape
        batch_size     = original_shape[0]

        sqrt_alpha = self.sqrt_alpha_cum_prod.to(xt.device)[t].reshape(batch_size)
        sqrt_one_m = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_m = sqrt_one_m.unsqueeze(-1)

        x0_pred    = torch.clamp(sqrt_alpha * xt - sqrt_one_m * v_pred, -1., 1.)
        noise_pred = sqrt_one_m * xt + sqrt_alpha * v_pred

        return x0_pred, noise_pred

    def sample_prev_timestep_v(self, xt, v_pred, t):
        """
        DDPM reverse step using v-prediction.

        Args:
            xt     (torch.Tensor): Noisy image at timestep t, shape (B, C, H, W).
            v_pred (torch.Tensor): Model's v prediction, same shape.
            t      (int | torch.Tensor): Current timestep.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (x_{t-1}, x0_pred).
        """
        if isinstance(t, int):
            t = torch.tensor([t] * xt.shape[0], device=xt.device, dtype=torch.long)
        elif t.dim() == 0:
            t = t.repeat(xt.shape[0])

        x0_pred, noise_pred = self.predict_x0_and_noise_from_v(xt, v_pred, t)

        mean = xt - (self.betas.to(xt.device)[t][0:1] * noise_pred) / \
               self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t][0:1]
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t][0:1])

        if t[0] == 0:
            return mean, x0_pred

        variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1][0:1]) / \
                   (1.0 - self.alpha_cum_prod.to(xt.device)[t][0:1])
        variance = variance * self.betas.to(xt.device)[t][0:1]
        sigma    = variance ** 0.5

        return mean + sigma * torch.randn_like(xt), x0_pred

    def sample_prev_timestep_v_ddim(self, xt, v_pred, t, prev_t=None, eta=0.0):
        """
        DDIM reverse step using v-prediction.

        Args:
            xt     (torch.Tensor): Noisy image at timestep t.
            v_pred (torch.Tensor): Model's v prediction.
            t      (int | torch.Tensor): Current timestep.
            prev_t (int | torch.Tensor | None): Previous timestep; defaults to t-1.
            eta    (float): Stochasticity factor (0 = fully deterministic DDIM).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (x_{prev_t}, x0_pred).
        """
        if isinstance(t, int):
            t = torch.tensor([t] * xt.shape[0], device=xt.device, dtype=torch.long)
        elif t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if prev_t is None:
            prev_t = t - 1
        if isinstance(prev_t, int):
            prev_t = torch.tensor([prev_t] * xt.shape[0], device=xt.device, dtype=torch.long)

        x0_pred, noise_pred = self.predict_x0_and_noise_from_v(xt, v_pred, t)

        if prev_t[0] < 0:
            return x0_pred, x0_pred

        alpha_t    = self.alpha_cum_prod.to(xt.device)[t][0:1]
        alpha_prev = self.alpha_cum_prod.to(xt.device)[prev_t][0:1]

        x_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        if eta > 0:
            variance = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            sigma    = torch.sqrt(variance) * eta
            x_prev   = x_prev + sigma * torch.randn_like(xt)

        return x_prev, x0_pred

    def get_snr_weight(self, t):
        """
        Returns SNR-based loss weights for the given timesteps.

        Uses the formula weight = 1 / sqrt(SNR + 1), which down-weights
        high-SNR (low-noise) timesteps relative to uniform weighting.

        Args:
            t (torch.Tensor): Timestep indices.

        Returns:
            torch.Tensor: Per-sample weights of the same shape as t.
        """
        snr = self.alpha_cum_prod[t] / (1 - self.alpha_cum_prod[t])
        return 1.0 / torch.sqrt(snr + 1)

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Standard DDPM reverse step using direct noise prediction (epsilon-prediction).

        Kept for compatibility with older checkpoints trained without v-prediction.

        Args:
            xt         (torch.Tensor): Noisy image at timestep t.
            noise_pred (torch.Tensor): Model's noise prediction.
            t          (int | torch.Tensor): Current timestep.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (x_{t-1}, x0_pred).
        """
        x0 = ((xt - self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = (xt - self.betas.to(xt.device)[t] * noise_pred /
                self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0

        variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / \
                   (1.0 - self.alpha_cum_prod.to(xt.device)[t])
        variance = variance * self.betas.to(xt.device)[t]
        sigma    = variance ** 0.5

        return mean + sigma * torch.randn_like(xt), x0
