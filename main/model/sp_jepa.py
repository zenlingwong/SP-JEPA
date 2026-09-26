"""Fixed-basis, increment-predicting JEPA world model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SPJEPASpec:
    profile: str
    channels: int
    height: int
    width: int
    history: int
    horizon: int
    rank: int = 32
    event_logits: int = 4
    event_condition: int = 4
    patch: int = 4
    d_h: int = 128
    readout: str = "linear"
    analytic_output: str = "increment"
    coordinate_basis: str = "fixed"

    @classmethod
    def ocean(cls):
        return cls("ocean_factual", 5, 90, 180, 12, 3, 32, 13, 12, 6, 128, "linear", "increment", "ocean_train_eof")

    @classmethod
    def paired(cls):
        return cls("paired_transport_exchange", 2, 16, 16, 6, 6, 32, 4, 4, 4, 128, "linear", "increment", "paired_lattice_gaussian")


class EncoderBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        q = self.n1(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class FrameEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        p, d = spec.patch, spec.d_h
        self.patch = nn.Conv2d(spec.channels + spec.channels + 4, d, p, p)
        n = (spec.height // p) * (spec.width // p)
        self.register_buffer("position", self._position(spec.height // p, spec.width // p, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.calendar = nn.Linear(2, d)
        self.blocks = nn.ModuleList([EncoderBlock(d) for _ in range(4)])
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        assert self.position.shape == (1, n, d)

    @staticmethod
    def _position(h, w, d):
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        xy = torch.stack((yy, xx), -1).reshape(-1, 2)
        f = torch.arange(max(1, d // 4), dtype=torch.float32)
        f = torch.exp(-torch.log(torch.tensor(10000.0)) * f / max(1, d // 4 - 1))
        pos = torch.cat([torch.sin(xy[:, :1] * f), torch.cos(xy[:, :1] * f),
                         torch.sin(xy[:, 1:] * f), torch.cos(xy[:, 1:] * f)], -1)
        return F.pad(pos[:, :d], (0, max(0, d - pos.shape[1])))[None]

    def forward(self, x, valid, geometry, calendar):
        b, length, channels, height, width = x.shape
        if geometry.ndim == 3:
            geometry = geometry[None].expand(b, -1, -1, -1)
        geom = geometry[:, None].expand(-1, length, -1, -1, -1)
        inp = torch.cat((torch.where(valid, x, torch.zeros_like(x)), valid.float(), geom), 2).flatten(0, 1)
        tokens = self.patch(inp).flatten(2).transpose(1, 2) + self.position
        cls = self.cls.expand(b * length, -1, -1) + self.calendar(calendar.flatten(0, 1))[:, None]
        tokens = torch.cat((cls, tokens), 1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.projector(tokens[:, 0]).reshape(b, length, -1)


class ConditionalCausalBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, gamma, beta):
        q = (1 + gamma) * self.n1(x) + beta
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        x = x + self.attn(q, q, q, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp((1 + gamma) * self.n2(x) + beta)


class SPJEPA(nn.Module):
    def __init__(self, spec, dictionary, weight, event_feedback=True, event_core_gradient=True):
        super().__init__()
        self.spec = spec
        self.event_feedback = bool(event_feedback)
        self.event_core_gradient = bool(event_core_gradient)
        d, rank = spec.d_h, spec.rank
        self.encoder = FrameEncoder(spec)
        self.a_in = nn.Linear(rank, d)
        self.cal_in = nn.Linear(2, d)
        self.time_pos = nn.Parameter(torch.zeros(1, spec.history, d))
        self.blocks = nn.ModuleList([ConditionalCausalBlock(d) for _ in range(4)])
        self.event_head = nn.Sequential(nn.Linear(spec.history * (d + rank), 128), nn.GELU(), nn.Linear(128, spec.event_logits))
        self.event_proj = nn.Linear(spec.event_condition, 2 * d)
        self.calendar_proj = nn.Linear(2, 2 * d)
        nn.init.zeros_(self.event_proj.weight); nn.init.zeros_(self.event_proj.bias)
        nn.init.zeros_(self.calendar_proj.weight); nn.init.zeros_(self.calendar_proj.bias)
        self.h_out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.a_out = nn.Linear(d, rank)
        nn.init.zeros_(self.a_out.weight); nn.init.zeros_(self.a_out.bias)
        self.decoder = nn.Linear(d, spec.channels * spec.height * spec.width)

        # The original model constructed an unused mask head here. Its parameters
        # were frozen and excluded from optimization, but construction advanced the
        # RNG before the second-stage event head was initialized.
        mask_width = 2 * spec.history * (d + rank) + 5
        _unused_mask_head = nn.Sequential(nn.Linear(mask_width, 128), nn.GELU(),
                                          nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        del _unused_mask_head
        self._init_dictionary(dictionary, weight)

    def _init_dictionary(self, dictionary, weight):
        n, rank = self.spec.channels * self.spec.height * self.spec.width, self.spec.rank
        if dictionary is None or tuple(dictionary.shape) != (n, rank):
            raise ValueError(f"dictionary must have shape [{n}, {rank}]")
        A64 = dictionary.detach().double().cpu()
        w = torch.ones(n, dtype=torch.float64) if weight is None else weight.detach().double().cpu().reshape(-1)
        gram64 = A64.T @ (w[:, None] * A64)
        if torch.linalg.matrix_rank(gram64) != rank:
            raise ValueError("dictionary is not full rank")
        ca64 = torch.linalg.solve(gram64, A64.T * w[None])
        self.register_buffer("A", A64.float())
        self.register_buffer("gram", gram64.float())
        self.register_buffer("C_A", ca64.float())

    def project_q(self, x):
        flat = x.flatten(-3)
        return (flat - (flat @ self.C_A.T) @ self.A.T).reshape_as(x)

    def coefficients(self, x):
        return x.flatten(-3) @ self.C_A.T

    def encode(self, x, valid, geometry, calendar):
        qx = self.project_q(x)
        h = self.encoder(qx, valid, geometry, calendar)
        return {"a": self.coefficients(x), "h": h}

    def action_coefficients(self, action):
        b = action["b"] if isinstance(action, dict) else action
        if b.shape[-1] == self.spec.rank:
            return b
        if self.spec.coordinate_basis == "paired_lattice_gaussian" and b.shape[-1] == 4:
            lifted = b.new_zeros(*b.shape[:-1], 32)
            lifted[..., [0, 8, 2, 10]] = b
            return lifted
        raise ValueError("action coefficient width mismatch")

    def edit(self, state, action):
        out = {key: value.clone() for key, value in state.items()}
        out["a"][:, -1] = out["a"][:, -1] + self.action_coefficients(action)
        return out

    def _events(self, state, detach_core=False):
        z = torch.cat((state["a"], state["h"]), -1)
        raw = self.event_head((z.detach() if detach_core else z).flatten(1))
        if self.spec.profile == "ocean_factual":
            pairs = raw[:, :10].reshape(-1, 5, 2).sigmoid().sort(-1).values.flatten(1)
            cond = torch.cat((pairs, raw[:, 10:].softmax(-1)[:, [2, 0]]), -1)
        else:
            pairs = raw.reshape(-1, 2, 2).sigmoid().sort(-1).values
            cond = pairs.flatten(1)
            raw = cond
        return raw, cond

    def read_events(self, state, detach_core=None):
        return self._events(state, not self.event_core_gradient if detach_core is None else detach_core)[0]

    def step(self, state, calendar_window, next_calendar):
        raw, cond = self._events(state, detach_core=False)
        cond = cond.detach()
        if not self.event_feedback:
            cond = torch.zeros_like(cond)
        x = state["h"] + self.cal_in(calendar_window) + self.time_pos
        x = x + self.a_in(state["a"])
        gb = self.event_proj(cond) + self.calendar_proj(next_calendar)
        gamma, beta = gb.chunk(2, -1); gamma = gamma[:, None]; beta = beta[:, None]
        for block in self.blocks:
            x = block(x, gamma, beta)
        last = x[:, -1]
        hnew = self.h_out(last)
        anew = state["a"][:, -1] + self.a_out(last)
        out = {"a": torch.cat((state["a"][:, 1:], anew[:, None]), 1),
               "h": torch.cat((state["h"][:, 1:], hnew[:, None]), 1)}
        cal = torch.cat((calendar_window[:, 1:], next_calendar[:, None]), 1)
        return {"a": anew, "h": hnew, "state": out, "calendar": cal,
                "event_condition": cond, "event_raw": raw}

    def rollout(self, state, calendar_window, future_calendar, K=None):
        K = future_calendar.shape[1] if K is None else K
        current = {key: value for key, value in state.items()}
        cal, frames, events = calendar_window, [], []
        for k in range(K):
            result = self.step(current, cal, future_calendar[:, k])
            frames.append({"a": result["a"], "h": result["h"]})
            current, cal = result["state"], result["calendar"]
            events.append(self.read_events(current))
        return {"a": torch.stack([z["a"] for z in frames], 1),
                "h": torch.stack([z["h"] for z in frames], 1),
                "events": torch.stack(events, 1), "state": current, "calendar": cal}

    def read_fields(self, z, geometry=None):
        h, a = z["h"], z["a"]
        lead_shape = h.shape[:-1]
        q = self.decoder(h.reshape(-1, self.spec.d_h)).reshape(*lead_shape, self.spec.channels, self.spec.height, self.spec.width)
        q = self.project_q(q)
        return q + (a @ self.A.T).reshape(*lead_shape, self.spec.channels, self.spec.height, self.spec.width)


def sigreg_v2(h, directions=1024, generator=None):
    """SIGReg over eight independent factual units."""
    if h.shape[0] != 8:
        raise ValueError("SIGReg requires batch size 8")
    d = h.shape[-1]
    z = h.reshape(h.shape[0], -1, d).transpose(0, 1)
    A = torch.randn(d, directions, device=h.device, dtype=h.dtype, generator=generator)
    A = F.normalize(A, dim=0)
    s = torch.linspace(0, 3, 17, device=h.device, dtype=h.dtype)
    phi = torch.exp(-s.square() / 2); ds = 3 / 16
    w = 2 * ds * phi; w[[0, -1]] = ds * phi[[0, -1]]
    xt = torch.einsum("nbd,dp->nbp", z, A)[..., None] * s
    err = (xt.cos().mean(1) - phi).square() + xt.sin().mean(1).square()
    return (h.shape[0] * (err * w).sum(-1)).mean()
