"""Ocean event readout and fusion for the released two-stage model."""

import math

import torch
from torch import nn

from .sp_jepa import SPJEPA


def calendar_month(calendar):
    return (torch.atan2(calendar[..., 0], calendar[..., 1]) * 6 / math.pi).round().long() % 12


class OceanSPJEPA(SPJEPA):
    def __init__(self, spec, dictionary, weight, table):
        super().__init__(spec, dictionary, weight, event_feedback=False, event_core_gradient=False)
        table = torch.as_tensor(table, dtype=torch.float32)
        anchor = torch.cat((torch.logit(table[:, :10].clamp(1e-3, 1 - 1e-3)),
                            table[:, 10:].clamp_min(1e-3).log()), -1)
        self.register_buffer("event_anchor", anchor)
        self.event_residual = nn.Sequential(nn.Linear(spec.history * (spec.d_h + spec.rank), 128),
                                            nn.GELU(), nn.Linear(128, spec.event_logits))
        nn.init.zeros_(self.event_residual[2].weight)
        nn.init.zeros_(self.event_residual[2].bias)
        self.register_buffer("event_gate", torch.ones(spec.event_logits))

    def add_fusion_layer(self):
        # Constructed after loading stage 1, matching the frozen two-stage script.
        self.ev_obs = nn.Linear(19, self.spec.d_h)
        nn.init.zeros_(self.ev_obs.weight)
        nn.init.zeros_(self.ev_obs.bias)

    def fuse(self, h, xi):
        return h + self.ev_obs(xi)

    def _events(self, state, detach_core=False):
        z = torch.cat((state["a"], state["h"]), -1)
        z = z.detach() if detach_core else z
        raw = self.event_anchor[calendar_month(state["cal"][:, -1])] + self.event_residual(z.flatten(1)) * self.event_gate
        pairs = raw[:, :10].reshape(-1, 5, 2).sigmoid().sort(-1).values.flatten(1)
        return raw, torch.cat((pairs, raw[:, 10:].softmax(-1)[:, [2, 0]]), -1)

    def step(self, state, calendar_window, next_calendar):
        result = super().step(state, calendar_window, next_calendar)
        result["state"]["cal"] = result["calendar"]
        return result
