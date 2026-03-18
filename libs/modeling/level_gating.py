"""
Adaptive Level Gating (Method B) for SnAG.

Drop this file into libs/modeling/level_gating.py

Per-sample, per-level binary gating via Gumbel-Sigmoid.
Input:  pooled FPN features + pooled text features (all detached)
Output: gate values g_l in (0,1) for each pyramid level
"""

import torch
import torch.nn as nn


class LevelGatingNetwork(nn.Module):

    def __init__(
        self,
        num_levels,         # number of FPN pyramid levels (6 for charades i3d)
        fpn_dim,            # channel dim of FPN features (256)
        text_dim,           # channel dim of text features (128)
        hidden_dim=128,     # MLP hidden size
        init_bias=2.0,      # positive = all levels start ON
        tau=2.0,            # Gumbel-Sigmoid temperature
        tau_min=0.5,        # minimum temperature after annealing
    ):
        super().__init__()
        self.num_levels = num_levels
        self.tau = tau
        self.tau_min = tau_min

        input_dim = num_levels * fpn_dim + text_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_levels),
        )
        # bias positive so all levels start active
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, fpn, fpn_masks, text, text_masks, text_size=None):
        """
        Args:
            fpn:        tuple of (bs_vid, C, T_l) — encoder FPN features
            fpn_masks:  tuple of (bs_vid, 1, T_l) — encoder FPN masks
            text:       (bs_query, C_text, T_text) — encoded text features
            text_masks: (bs_query, T_text) or (bs_query, 1, T_text)
            text_size:  (bs_vid,) — number of queries per video, or None

        Returns:
            gates: (bs_query, num_levels) — soft (train) or hard (eval)
        """
        # --- pool FPN per level: (bs_vid, C) each ---
        pooled = []
        for f, m in zip(fpn, fpn_masks):
            # f: (bs_vid, C, T), m: (bs_vid, 1, T)
            m_f = m.float()
            denom = m_f.sum(-1).clamp(min=1)        # (bs_vid, 1)
            p = (f * m_f).sum(-1) / denom            # (bs_vid, C)
            pooled.append(p)
        h_vid = torch.cat(pooled, dim=-1)             # (bs_vid, num_levels * C)

        # --- pool text: (bs_query, C_text) ---
        if text_masks.ndim == 2:
            tm = text_masks.unsqueeze(1).float()      # (bs_q, 1, T_text)
        else:
            tm = text_masks.float()                    # (bs_q, 1, T_text)
        denom = tm.sum(-1).clamp(min=1)               # (bs_q, 1)
        h_text = (text * tm).sum(-1) / denom           # (bs_q, C_text)

        # --- expand video summaries to match queries ---
        if text_size is not None and h_vid.size(0) != h_text.size(0):
            h_vid = h_vid.repeat_interleave(text_size, dim=0)

        # --- detach: gate is a pure routing function ---
        z = torch.cat([h_vid, h_text], dim=-1).detach()

        logits = self.mlp(z)                           # (bs_query, num_levels)

        if self.training:
            # Gumbel-Sigmoid: differentiable soft binary gate
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            noise = torch.log(u / (1 - u))            # logistic noise
            gates = torch.sigmoid((logits + noise) / self.tau)
        else:
            # hard binary gate at inference
            gates = (logits > 0).float()

        return gates

    def set_tau(self, tau):
        self.tau = max(tau, self.tau_min)