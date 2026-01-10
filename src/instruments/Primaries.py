from abc import abstractmethod

import torch
import numpy as np
from instruments.Instruments import Instrument


class Primary(Instrument):

    @abstractmethod
    def simulate(self, P, T) -> torch.Tensor:
        # returns P x T tensor of primary paths
        pass

    def value(self, primary_path):
        return primary_path

    def primary(self) -> Instrument:
        return self

    def delattr(self, primary_path):
        return torch.ones_like(primary_path)

class GeometricBrownianStock_outdated(Primary):

    def __init__(self, S0, mu, sigma):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def name(self):
        return f"Geometric Brownian Stock with S0={self.S0}, mu={self.mu}, sigma={self.sigma}"

    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        # the first one should be S0
        return torch.cat([
            self.S0 * torch.ones(P, 1),
            self.S0 * torch.exp(torch.cumsum((self.mu - 0.5 * self.sigma ** 2) * torch.ones(P, T - 1) + self.sigma * torch.randn(P, T - 1), dim=1))
        ], dim=1)

import torch

class GeometricBrownianStock(Primary):
    """
    Simulates GBM paths and returns a (P, T) price matrix, where:
      - P = number of paths
      - T = number of columns in the returned matrix (including S0 at t=0)

    Convention:
      - total horizon = horizon (float, e.g. 1.0 year)
      - number of steps = T-1
      - dt = horizon / (T-1)
    """

    def __init__(self, S0: float, mu: float, sigma: float, horizon: float = 1.0, seed: int = 0, device=None, dtype=torch.float32):
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.horizon = float(horizon)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

    def name(self):
        return f"Geometric Brownian Stock with S0={self.S0}, mu={self.mu}, sigma={self.sigma}, horizon={self.horizon}"

    def simulate(self, P: int, T: int) -> torch.Tensor:
        """
        Returns:
          S: torch.Tensor of shape (P, T), with S[:,0] = S0.
        """
        # device handling
        device = self.device if self.device is not None else torch.device("cpu")

        # reproducibility (optional; remove if you want randomness each call)
        torch.manual_seed(self.seed)

        if T <= 0:
            raise ValueError("T must be >= 1.")
        if P <= 0:
            raise ValueError("P must be >= 1.")
        if T == 1:
            return torch.full((P, 1), self.S0, device=device, dtype=self.dtype)

        n_steps = T - 1
        dt = self.horizon / n_steps

        z = torch.randn(P, n_steps, device=device, dtype=self.dtype)

        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * torch.sqrt(torch.tensor(dt, device=device, dtype=self.dtype)) * z

        log_returns = drift + diffusion
        log_S = torch.cumsum(log_returns, dim=1)

        # prepend log S_0 = 0 so that S_0 * exp(0) = S_0
        log_S = torch.cat([torch.zeros(P, 1, device=device, dtype=self.dtype), log_S], dim=1)

        return self.S0 * torch.exp(log_S) * 30

class HestonStock(Primary):
    """
    Heston stochastic volatility model.

    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW1_t
    dv_t = kappa*(theta - v_t) dt + xi*sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho

    Discretization:
      - v: full-truncation Euler (keeps variance nonnegative-ish)
      - S: log-Euler using v_t (or v_{t+dt} if you prefer)
    """

    def __init__(self, S0, mu, v0, kappa, theta, xi, rho, dt=1.0, eps=1e-12, device=None, dtype=torch.float32):
        self.S0 = float(S0)
        self.mu = float(mu)

        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)

        self.dt = float(dt)
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype

        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must be in (-1, 1).")
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0 or self.v0 < 0:
            raise ValueError("Require kappa>0, theta>0, xi>0, v0>=0.")

    def name(self):
        return (f"Heston Stock with S0={self.S0}, mu={self.mu}, v0={self.v0}, "
                f"kappa={self.kappa}, theta={self.theta}, xi={self.xi}, rho={self.rho}, dt={self.dt}")

    @torch.no_grad()
    def simulate(self, P, T):
        """
        Returns:
          S_paths: torch.Tensor of shape (P, T) with S_paths[:,0]=S0
        """
        device = self.device if self.device is not None else "cpu"
        dtype = self.dtype

        P = int(P)
        T = int(T)
        if T < 1:
            raise ValueError("T must be >= 1.")

        dt = self.dt
        sqrt_dt = dt ** 0.5

        # Allocate
        S = torch.empty((P, T), device=device, dtype=dtype)
        v = torch.empty((P, T), device=device, dtype=dtype)

        # Initial
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Pre-generate independent normals
        Z1 = torch.randn((P, T - 1), device=device, dtype=dtype)
        Z2 = torch.randn((P, T - 1), device=device, dtype=dtype)

        # Correlate: dW2 = rho*dW1 + sqrt(1-rho^2)*dW_perp
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (self.rho * Z1 + (1.0 - self.rho**2) ** 0.5 * Z2)

        # Iterate
        for t in range(T - 1):
            vt = v[:, t]

            # Full truncation: use vt_pos in diffusion; update v then truncate
            vt_pos = torch.clamp(vt, min=0.0)
            sqrt_vt = torch.sqrt(torch.clamp(vt_pos, min=self.eps))

            # Variance update
            v_next = vt + self.kappa * (self.theta - vt_pos) * dt + self.xi * sqrt_vt * dW2[:, t]
            v_next = torch.clamp(v_next, min=0.0)  # enforce nonnegativity
            v[:, t + 1] = v_next

            # Stock update (log-Euler)
            # Use vt_pos (or v_next) for instantaneous variance in this step; both are used in literature.
            S[:, t + 1] = S[:, t] * torch.exp((self.mu - 0.5 * vt_pos) * dt + sqrt_vt * dW1[:, t])

        return S


class HestonStock_oudated(Primary):
    # only returns stock price, not variance
    # as this would need some editing

    def __init__(self, S0, V0, mu, kappa, theta, xi, rho):
        self.S0 = S0
        self.V0 = V0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def name(self):
        return f"Heston Stock with S0={self.S0}, V0={self.V0}, mu={self.mu}, kappa={self.kappa}, theta={self.theta}, xi={self.xi}, rho={self.rho}"

    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        # the first one should be S0
        # the second one should be V0
        S = torch.zeros(P, T)
        V = torch.zeros(P, T)

        #initial values (P x 1)
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        step_size = 1/T

        #generate increments (normal distributed with mean 0, variance sqrt(step_size))
        stock_increments = torch.randn(P, T - 1) * np.sqrt(step_size)
        variance_increments = torch.randn(P, T - 1) * np.sqrt(step_size)

        for t in range(1, T):
            #outdated: S[:, t] = S[:, t-1] + self.mu * S[:, t-1] * step_size + torch.sqrt(torch.clamp(V[:,t-1], min=0)) * S[:, t-1] * stock_increments[:, t-1]
            #outdated: V[:, t] = V[:, t-1] + self.kappa * (self.theta - torch.clamp(V[:,t-1], min=0)) * step_size + self.xi * torch.sqrt(torch.clamp(V[:,t-1], min=0)) * (self.rho * variance_increments[:, t-1] + np.sqrt(1 - self.rho**2) * stock_increments[:, t-1])
            S[:, t] = S[:, t-1] + self.mu * S[:, t-1] * step_size + torch.sqrt(torch.clamp(V[:,t-1], min=0)) * S[:, t-1] * stock_increments[:, t-1]
            V[:, t] = V[:, t-1] + self.kappa * (self.theta - torch.clamp(V[:,t-1], min=0)) * step_size + self.xi * torch.sqrt(torch.clamp(V[:,t-1], min=0)) * (self.rho * variance_increments[:, t-1] + np.sqrt(1 - self.rho**2) * stock_increments[:, t-1])

        return S
