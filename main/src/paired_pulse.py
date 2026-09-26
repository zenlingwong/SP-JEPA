"""Paired Transport–Exchange truth: periodic two-field advection, diffusion, exchange.

The solver and independent random streams follow the registered nonlinear
paired profile. Hidden mechanism parameters are data provenance only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


PROFILE = "paired_nonlinear_exchange_v1"
SPLIT_CODE = {"train": 0x54524149, "validation-selection": 0x5653454C,
              "validation-calibration": 0x5643414C, "test": 0x54455354}
SHAPE = {"train": (32, 16), "validation-selection": (4, 16),
         "validation-calibration": (4, 16), "test": (16, 16)}
DT = 1 / 16


@dataclass(frozen=True)
class Mechanism:
    velocity: tuple[float, float]
    diffusivity: tuple[float, float]
    exchange: tuple[float, float]
    gamma: float = .2


@dataclass(frozen=True)
class Action:
    index: int
    center_index: int
    sign: int
    amplitude: float
    delta: torch.Tensor


def _rng(split: str, namespace: int, *identity: int) -> np.random.Generator:
    entropy = [0x4E4F4E4C, SPLIT_CODE[split], namespace, *map(int, identity)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def mechanism_parameters(split: str, mechanism_id: int) -> Mechanism:
    rng = _rng(split, 0x4D454348, mechanism_id)
    return Mechanism(tuple(map(float, rng.uniform(-.35, .35, 2))),
                     tuple(map(float, rng.uniform(.015, .060, 2))),
                     tuple(map(float, rng.uniform(.020, .120, 2))))


def initial_state(split: str, mechanism_id: int, initial_state_id: int) -> torch.Tensor:
    rng = _rng(split, 0x494E4954, mechanism_id, initial_state_id)
    y = 2 * np.pi * np.arange(16) / 16
    yy, xx = np.meshgrid(y, y, indexing="ij")
    state = []
    for low, high in ((.90, 1.20), (.70, 1.00)):
        field = np.full_like(yy, rng.uniform(low, high), dtype=np.float64)
        for _ in range(4):
            kx, ky = rng.integers(0, 4, size=2)
            if kx == 0 and ky == 0:
                kx = 1
            amplitude = rng.uniform(-.06, .06)
            field += amplitude * np.cos(kx * xx + ky * yy + rng.uniform(0, 2 * np.pi))
        state.append(field)
    return torch.from_numpy(np.stack(state))


def action_bank() -> tuple[Action, ...]:
    y = 2 * math.pi * torch.arange(16, dtype=torch.float64) / 16
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    actions = []
    for center, (cy, cx) in enumerate(((0., 0.), (math.pi, 0.), (0., math.pi), (math.pi, math.pi))):
        dy = torch.minimum((yy - cy).abs(), 2 * math.pi - (yy - cy).abs())
        dx = torch.minimum((xx - cx).abs(), 2 * math.pi - (xx - cx).abs())
        template = torch.exp(-(dx.square() + dy.square()) / (2 * .6 ** 2))
        for amplitude in (.05, .10):
            for sign in (-1, 1):
                delta = torch.zeros(2, 16, 16, dtype=torch.float64)
                delta[0] = sign * amplitude * template
                actions.append(Action(len(actions), center, sign, amplitude, delta))
    return tuple(actions)


def _spectral_operator(mechanism: Mechanism, dt: float, device):
    vx, vy = mechanism.velocity
    k1, k2 = mechanism.diffusivity
    a, b = mechanism.exchange
    ky = torch.fft.fftfreq(16, d=1 / 16, dtype=torch.float64, device=device)
    kx = ky.clone()
    dky, dkx = ky.clone(), kx.clone()
    dky[8] = 0
    dkx[8] = 0
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    dky, dkx = torch.meshgrid(dky, dkx, indexing="ij")
    k2grid = kx.square() + ky.square()
    generator = torch.empty(16, 16, 2, 2, dtype=torch.float64, device=device)
    generator[..., 0, 0] = -k1 * k2grid - a
    generator[..., 0, 1] = b
    generator[..., 1, 0] = a
    generator[..., 1, 1] = -k2 * k2grid - b
    return torch.matrix_exp(dt * generator), torch.exp(-1j * dt * (vx * dkx + vy * dky))


def _apply_operator(state, mixing, advection):
    modes = torch.fft.fft2(state, dim=(-2, -1)).movedim(-3, -1)
    mixed = torch.einsum("yxij,...yxj->...yxi", mixing.to(modes.dtype), modes)
    return torch.fft.ifft2((mixed * advection[..., None]).movedim(-1, -3), dim=(-2, -1)).real


def _reaction_step(state, dt: float):
    x1, x2 = state.unbind(dim=-3)
    total = x1 + x2
    decay = torch.exp(-.2 * total * dt)
    denominator = x2 + x1 * decay
    next_x1 = torch.where(total == 0, torch.zeros_like(total), total * x1 * decay / denominator)
    return torch.stack((next_x1, total - next_x1), dim=-3)


def evolve(state: torch.Tensor, steps: int, mechanism: Mechanism, dt: float = DT) -> torch.Tensor:
    """Each recording interval uses 1/dt symmetric linear/reaction substeps."""
    current = torch.as_tensor(state, dtype=torch.float64)
    substeps = int(round(1 / dt))
    if not math.isclose(substeps * dt, 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("dt must divide one recording interval")
    half_mixing, half_advection = _spectral_operator(mechanism, dt / 2, current.device)
    result = []
    for _ in range(steps):
        for _ in range(substeps):
            current = _apply_operator(current, half_mixing, half_advection)
            current = _reaction_step(current, dt)
            current = _apply_operator(current, half_mixing, half_advection)
        result.append(current)
    return torch.stack(result)


def generate_root(split: str, mechanism_id: int, initial_id: int, dt: float = DT):
    """Return native float64 history and factual future from one mechanism/root."""
    mechanism = mechanism_parameters(split, mechanism_id)
    initial = initial_state(split, mechanism_id, initial_id)
    history = torch.cat((initial[None], evolve(initial, 5, mechanism, dt)), 0)
    factual = evolve(history[-1], 6, mechanism, dt)
    return history, factual


def changed_future(history: torch.Tensor, mechanism: Mechanism, action: Action, dt: float = DT):
    if torch.any(history[-1] + action.delta <= 0):
        raise ValueError("action leaves positive domain")
    return evolve(history[-1] + action.delta, 6, mechanism, dt)


def fit_statistics(history: torch.Tensor, factual: torch.Tensor) -> dict[str, torch.Tensor]:
    """All 512 TRAIN factual roots, 6 history + 6 future frames each."""
    timeline = torch.cat((history, factual), dim=1).double()
    mean = timeline.mean((0, 1, 3, 4))
    std = timeline.std((0, 1, 3, 4), unbiased=False)
    thresholds = torch.quantile(timeline.movedim(2, -1).reshape(-1, 2), .90, dim=0)
    proportions = (timeline > thresholds[None, None, :, None, None]).double().mean((3, 4))
    scale = proportions.reshape(-1, 2).std(0, unbiased=False).clamp_min(.05)
    return {"mean": mean, "std": std, "event_threshold": thresholds, "event_scale": scale}


def event_bounds(values: torch.Tensor, thresholds: torch.Tensor, epsilon) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values, dtype=torch.float64)
    thresholds = torch.as_tensor(thresholds, dtype=torch.float64)
    epsilon = torch.as_tensor(epsilon, dtype=torch.float64)
    view = (1,) * (values.ndim - 3) + (2, 1, 1)
    lower = (values > (thresholds + epsilon).view(view)).double().mean((-2, -1))
    upper = (values > (thresholds - epsilon).view(view)).double().mean((-2, -1))
    return torch.stack((lower, upper), -1), lower == upper
