"""
Adaptive Level Gating (Method B) for SnAG.

Place at: libs/modeling/level_gating.py

Gate input:  pooled FPN features + pooled text features (ALL detached)
Gate output: per-query, per-level soft (train) or hard (eval) gates

    z = stopgrad([h_0, ..., h_5, h_text])
    logit_l = MLP(z)_l

    Training: g_l = sigmoid((logit_l + logistic_noise) / tau)
    Inference: g_l = 1 if logit_l > 0 else 0
"""

import torch
import torch.nn as nn


class LevelGatingNetwork(nn.Module):

    def __init__(
        self,
        num_levels,         # number of FPN pyramid levels (6)
        fpn_dim,            # channel dim of FPN features (256)
        text_dim,           # channel dim of text features (128)
        hidden_dim=128,     # MLP hidden size
        init_bias=0.0,      # sigmoid(0) = 0.5 → start uncertain
        tau=2.0,            # Gumbel-Sigmoid temperature (high = soft)
        tau_min=0.5,        # final temperature (low = near-hard)
    ):
        super().__init__()
        self.num_levels = num_levels
        self.tau = tau
        self.tau_min = tau_min

        # input: L pooled FPN vectors + 1 pooled text vector
        input_dim = num_levels * fpn_dim + text_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_levels),
        )
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, fpn, fpn_masks, text, text_masks, text_size=None):
        """
        Args:
            fpn:        tuple of (bs_vid, C, T_l)
            fpn_masks:  tuple of (bs_vid, 1, T_l)
            text:       (bs_query, C_text, T_text) — already encoded
            text_masks: (bs_query, 1, T_text) or (bs_query, T_text)
            text_size:  (bs_vid,) — queries per video, or None

        Returns:
            gates: (bs_query, num_levels)
        """
        # --- pool each FPN level: (bs_vid, C) ---
        pooled_fpn = []
        for f, m in zip(fpn, fpn_masks):
            m_f = m.float()                              # (bs_vid, 1, T)
            denom = m_f.sum(-1).clamp(min=1)             # (bs_vid, 1)
            p = (f * m_f).sum(-1) / denom                # (bs_vid, C)
            pooled_fpn.append(p)
        h_vid = torch.cat(pooled_fpn, dim=-1)            # (bs_vid, L * C)

        # --- pool text: (bs_query, C_text) ---
        if text_masks.ndim == 2:
            tm = text_masks.unsqueeze(1).float()          # (bs_q, 1, T_text)
        else:
            tm = text_masks.float()
        denom = tm.sum(-1).clamp(min=1)
        h_text = (text * tm).sum(-1) / denom              # (bs_q, C_text)

        # --- expand h_vid from bs_vid to bs_query ---
        if text_size is not None and h_vid.size(0) != h_text.size(0):
            h_vid = h_vid.repeat_interleave(text_size, dim=0)

        # --- stopgrad: gate is a pure routing function ---
        z = torch.cat([h_vid, h_text], dim=-1).detach()   # (bs_q, L*C + C_text)

        logits = self.mlp(z)                               # (bs_q, L)

        # if self.training:
        #     # Gumbel-Sigmoid
        #     u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        #     noise = torch.log(u / (1 - u))
        #     gates = torch.sigmoid((logits + noise) / self.tau)
        # else:
        #     # hard binary
        #     gates = (logits > 0).float()
        soft = torch.sigmoid(logits / self.tau)
    
        if self.training:
            hard = (soft > 0.5).float()
            # straight-through: use hard in forward, soft in backward
            gates = hard - soft.detach() + soft
            soft_gates = soft
        else:
            gates = (logits > 0).float()
            soft_gates = None

        return gates, soft_gates

    def set_tau(self, tau):
        self.tau = max(tau, self.tau_min)