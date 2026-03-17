from torch import nn
import torch
import math
import torch.nn.functional as F

HALTING_REGISTRY = {}

def register_halting(name):
    def decorator(cls):
        HALTING_REGISTRY[name] = cls
        return cls
    return decorator

def build_halting(cfg):
    """Build a halting module from config. Returns None if not specified."""
    if cfg is None:
        return None
    name = cfg['type']
    return HALTING_REGISTRY[name](**{k: v for k, v in cfg.items() if k != 'type'})


# ---- (B.2) Learned halting module ----

@register_halting("learned")
class LearnedHalting(nn.Module):
    """
    PonderNet-style learned halting.
    Projects hidden state to a scalar halt probability.
    """
    def __init__(self, d_model, prior_p=0.3):
        super().__init__()
        self.halt_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.prior_p = prior_p  # geometric prior parameter

    def forward(self, hidden, mask=None):
        """
        Args:
            hidden: (bs, t, d) or (n_active, t, d)
            mask:   (bs, t) or (n_active, t), optional
        Returns:
            halt_prob: (bs,) or (n_active,) in [0, 1]
        """
        if mask is not None:
            # mean-pool over valid positions
            hidden = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            hidden = hidden.mean(1)
        return torch.sigmoid(self.halt_proj(hidden).squeeze(-1))

    def compute_loss(self, all_halt_probs, iters_used, max_iters):
        """
        KL divergence between halting distribution and geometric prior.
        
        Args:
            all_halt_probs: list of (bs,) tensors, one per iteration
            iters_used: (bs,) int tensor
            max_iters: int
        Returns:
            kl_loss: scalar
        """
        # Build per-sample halting distribution: p(halt at step n)
        # p(n) = halt_prob[n] * prod_{k<n}(1 - halt_prob[k])
        eps = 1e-7
        log_survival = torch.zeros_like(all_halt_probs[0])
        log_p_halt = []

        for i, hp in enumerate(all_halt_probs):
            hp = hp.clamp(eps, 1 - eps)
            if i < len(all_halt_probs) - 1:
                log_p_halt.append(log_survival + torch.log(hp))
                log_survival = log_survival + torch.log(1 - hp)
            else:
                # last iteration: must halt (remainder mass)
                log_p_halt.append(log_survival)

        log_p_halt = torch.stack(log_p_halt, dim=0)  # (max_iters, bs)

        # Geometric prior: p(n) = (1-p)^(n-1) * p, truncated at max_iters
        n = torch.arange(1, max_iters + 1, device=log_p_halt.device).float()
        log_prior = (n - 1) * math.log(1 - self.prior_p) + math.log(self.prior_p)
        # truncate: renormalize
        log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
        log_prior = log_prior[:, None]  # (max_iters, 1)

        # KL(q || p) = sum q * (log q - log p)
        q = torch.exp(log_p_halt)
        kl = (q * (log_p_halt - log_prior)).sum(0).mean()
        return kl


# ---- (B.3) Parameter-free entropy-based halting ----
@register_halting("entropy")
class EntropyHalting(nn.Module):
    def __init__(self, threshold=1.0, patience=1, ema_momentum=0.99):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('num_updates', torch.tensor(0))
        self.ema_momentum = ema_momentum
        self._patience_counter = None

    def forward(self, hidden, mask=None, itr=0, attn_weights=None, sample_fn=None):
        """
        Args:
            hidden:      (bs, d, t) — unused, kept for interface consistency
            mask:        (bs, 1, t) or (bs, t)
            itr:         unused
            attn_weights: (bs, heads, t, t)
            sample_fn:   unused
        """
        assert attn_weights is not None, "Entropy halting requires attn_weights"
        entropy = self._attention_entropy(attn_weights, mask)

        if self.training:
            with torch.no_grad():
                self.num_updates += 1
                self.running_mean = (
                    self.ema_momentum * self.running_mean +
                    (1 - self.ema_momentum) * entropy.mean()
                )
                self.running_var = (
                    self.ema_momentum * self.running_var +
                    (1 - self.ema_momentum) * entropy.var()
                )

        adaptive_thresh = self.running_mean - self.threshold * (
            self.running_var.sqrt() + 1e-6
        )
        should_halt = entropy < adaptive_thresh

        if self._patience_counter is None or self._patience_counter.size(0) != hidden.size(0):
            self._patience_counter = torch.zeros(hidden.size(0), device=hidden.device)
        self._patience_counter = torch.where(
            should_halt,
            self._patience_counter + 1,
            torch.zeros_like(self._patience_counter),
        )
        halt_prob = (self._patience_counter >= self.patience).float()
        return halt_prob

    def reset_patience(self, bs, device):
        self._patience_counter = torch.zeros(bs, device=device)

    def _attention_entropy(self, attn_weights, mask=None):
        eps = 1e-8
        ent = -(attn_weights * (attn_weights + eps).log()).sum(-1)
        if mask is not None:
            m = mask.squeeze(1) if mask.ndim == 3 else mask
            ent = (ent * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp(min=1)
        else:
            ent = ent.mean(-1)
        return ent.mean(-1)

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0, requires_grad=False)
    
@register_halting("mc_dropout")
class MCDropoutHalting(nn.Module):
    def __init__(self, n_samples=5, threshold=0.5, ema_momentum=0.99, max_iters=8):
        super().__init__()
        self.n_samples = n_samples
        self.threshold = threshold  # fraction of initial variance
        self.ema_momentum = ema_momentum
        # track variance per iteration depth
        self.register_buffer('var_per_iter', torch.zeros(max_iters))
        self.register_buffer('var_iter0', torch.tensor(0.0))  # variance at first iteration
        self.register_buffer('initialized', torch.tensor(False))

    def forward_warmup(self, hidden, mask, itr, sample_fn=None, attn_weights=None):
        assert sample_fn is not None
        samples = torch.stack([sample_fn() for _ in range(self.n_samples)])
        pred_var = samples.var(dim=0).mean(dim=(1, 2))

        with torch.no_grad():
            momentum = self.ema_momentum if self.initialized else 0.0
            self.var_per_iter[itr] = (
                momentum * self.var_per_iter[itr] +
                (1 - momentum) * pred_var.mean()
            )
            if itr == 0:
                self.var_iter0 = self.var_per_iter[0]
            self.initialized.fill_(True)

    def forward(self, hidden, mask=None, itr=0, attn_weights=None, sample_fn=None):
        assert sample_fn is not None

        samples = torch.stack([sample_fn() for _ in range(self.n_samples)])
        pred_var = samples.var(dim=0).mean(dim=(1, 2))

        if self.training:
            with torch.no_grad():
                self.var_per_iter[itr] = (
                    self.ema_momentum * self.var_per_iter[itr] +
                    (1 - self.ema_momentum) * pred_var.mean()
                )

        halt_thresh = self.threshold * self.var_iter0
        halt_prob = (pred_var < halt_thresh).float()

        # store for logging
        self._last_var = pred_var.detach()
        self._last_thresh = halt_thresh.detach()
        self._last_itr = itr

        return halt_prob

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0, requires_grad=False)

@register_halting("calibrated")
class CalibratedHalting(nn.Module):
    """
    Two-phase halting:
      Phase 1 (warmup): Compute both MC dropout variance and prediction
          entropy at each iteration. Build a mapping from entropy to variance.
      Phase 2 (post-warmup): Use prediction entropy only, with thresholds
          calibrated against the variance signal from phase 1.
    
    This gives Bayesian-grounded thresholds with cheap runtime halting.
    """
    def __init__(
        self,
        n_samples=5,
        threshold=1.0,
        patience=1,
        max_iters=8,
        ema_momentum=0.99,
        calibration_mode='prediction_entropy',  # or 'attention_entropy'
    ):
        super().__init__()

        object.__setattr__(self, 'probe_fn', None)
        object.__setattr__(self, 'stem_fn', None)
        self.register_buffer('var_iter0', torch.zeros(max_iters))

        self.n_samples = n_samples
        self.threshold = threshold
        self.patience = patience
        self.max_iters = max_iters
        self.ema_momentum = ema_momentum
        self.calibration_mode = calibration_mode

        # --- calibration buffers (built during warmup) ---
        # per-iteration running stats for variance
        self.register_buffer(
            'var_running_mean', torch.zeros(max_iters)
        )
        self.register_buffer(
            'var_running_var', torch.ones(max_iters)
        )
        # per-iteration running stats for entropy
        self.register_buffer(
            'ent_running_mean', torch.zeros(max_iters)
        )
        self.register_buffer(
            'ent_running_var', torch.ones(max_iters)
        )
        # per-iteration correlation between variance and entropy
        self.register_buffer(
            'correlation', torch.zeros(max_iters)
        )
        # per-iteration: what entropy value corresponds to the
        # variance threshold? (the calibrated mapping)
        self.register_buffer(
            'calibrated_entropy_thresh', torch.zeros(max_iters)
        )
        self.register_buffer(
            'num_calibration_updates', torch.zeros(max_iters)
        )

        # running covariance terms for online correlation computation
        self.register_buffer('cov_sum_xy', torch.zeros(max_iters))
        self.register_buffer('cov_sum_x', torch.zeros(max_iters))
        self.register_buffer('cov_sum_y', torch.zeros(max_iters))
        self.register_buffer('cov_sum_x2', torch.zeros(max_iters))
        self.register_buffer('cov_sum_y2', torch.zeros(max_iters))
        self.register_buffer('cov_n', torch.zeros(max_iters))

        self._patience_counter = None


    def forward_warmup(self, hidden, mask=None, itr=0, sample_fn=None, attn_weights=None):
        """
        Warmup phase: compute both signals and update calibration.
        Does NOT make halting decisions (warmup uses fixed depth).
        
        Args:
            hidden:       (n_active, d, t)
            mask:         (n_active, 1, t)
            itr:          current iteration index
            sample_fn:    callable for MC dropout passes
            attn_weights: attention weights from current iteration
        """
        bs = hidden.size(0)

        # --- signal 1: MC dropout variance ---
        assert sample_fn is not None
        samples = torch.stack([
            sample_fn() for _ in range(self.n_samples)
        ])
        variance = samples.var(dim=0).mean(dim=(1, 2))

        # --- signal 2: cheap entropy ---
        if self.calibration_mode == 'prediction_entropy':
            assert self.probe_fn is not None
            with torch.no_grad():
                logits, _ = self.probe_fn([hidden], [mask])
                logits = torch.cat(logits, dim=1)  # (n_active, p)
            probs = torch.sigmoid(logits)
            point_entropy = self._binary_entropy(probs)
            point_mask = (logits != 0).float()
            entropy = (
                (point_entropy * point_mask).sum(-1) /
                point_mask.sum(-1).clamp(min=1)
            )  # (n_active,)
        else:
            # attention entropy
            assert attn_weights is not None
            entropy = self._attention_entropy(attn_weights, mask)

        # --- update per-iteration running stats ---
        with torch.no_grad():
            n = self.num_calibration_updates[itr]
            momentum = self.ema_momentum if n > 0 else 0.0

            self.var_running_mean[itr] = (
                momentum * self.var_running_mean[itr] +
                (1 - momentum) * variance.mean()
            )
            self.var_running_var[itr] = (
                momentum * self.var_running_var[itr] +
                (1 - momentum) * variance.var().clamp(min=1e-8)
            )
            self.ent_running_mean[itr] = (
                momentum * self.ent_running_mean[itr] +
                (1 - momentum) * entropy.mean()
            )
            self.ent_running_var[itr] = (
                momentum * self.ent_running_var[itr] +
                (1 - momentum) * entropy.var().clamp(min=1e-8)
            )

            # --- NEW: track iteration 0 variance for ratio-based threshold ---
            if itr == 0:
                self.var_iter0[0] = self.var_running_mean[0]

            # --- online correlation computation ---
            # accumulate sums for Pearson correlation
            self.cov_n[itr] += bs
            self.cov_sum_x[itr] += variance.sum()
            self.cov_sum_y[itr] += entropy.sum()
            self.cov_sum_x2[itr] += (variance ** 2).sum()
            self.cov_sum_y2[itr] += (entropy ** 2).sum()
            self.cov_sum_xy[itr] += (variance * entropy).sum()

            self.num_calibration_updates[itr] += 1

    def finalize_calibration(self):
        """
        Call once at the end of warmup.
        Computes per-iteration correlations and calibrated thresholds.
        """
        for itr in range(self.max_iters):
            n = self.cov_n[itr]
            if n < 2:
                continue

            # ratio-based variance threshold (same as MCDropoutHalting)
            var_thresh = self.threshold * self.var_iter0[0]

            # Pearson correlation
            mean_x = self.cov_sum_x[itr] / n
            mean_y = self.cov_sum_y[itr] / n
            var_x = self.cov_sum_x2[itr] / n - mean_x ** 2
            var_y = self.cov_sum_y2[itr] / n - mean_y ** 2
            cov_xy = self.cov_sum_xy[itr] / n - mean_x * mean_y

            denom = (var_x * var_y).sqrt().clamp(min=1e-8)
            self.correlation[itr] = cov_xy / denom

            # calibrated entropy threshold:
            # "what entropy corresponds to variance = threshold * var_iter0?"
            # Using linear mapping: entropy ≈ slope * variance + intercept
            if var_x.abs() > 1e-8:
                slope = cov_xy / var_x
                intercept = mean_y - slope * mean_x
                self.calibrated_entropy_thresh[itr] = slope * var_thresh + intercept
            else:
                # fallback: can't build linear mapping, use entropy ratio instead
                ent_mean = self.ent_running_mean[itr]
                ent_iter0 = self.ent_running_mean[0]
                self.calibrated_entropy_thresh[itr] = self.threshold * ent_iter0
                
    def forward(self, hidden, mask=None, itr=0, attn_weights=None, sample_fn=None):
        """
        Post-warmup: use cheap entropy signal with calibrated thresholds.
        
        Args:
            hidden:       (n_active, d, t)
            mask:         (n_active, 1, t) or (n_active, t)
            itr:          current iteration index
            attn_weights: attention weights (for attention_entropy mode)
        Returns:
            halt_prob: (n_active,)
        """
        bs = hidden.size(0)
        device = hidden.device

        if self.calibration_mode == 'prediction_entropy':
            assert self.probe_fn is not None
            with torch.no_grad():
                logits, _ = self.probe_fn([hidden], [mask])
                logits = torch.cat(logits, dim=1)
            probs = torch.sigmoid(logits)
            point_entropy = self._binary_entropy(probs)
            point_mask = (logits != 0).float()
            entropy = (
                (point_entropy * point_mask).sum(-1) /
                point_mask.sum(-1).clamp(min=1)
            )
        else:
            assert attn_weights is not None
            entropy = self._attention_entropy(attn_weights, mask)

        # use calibrated threshold for this iteration depth
        thresh = self.calibrated_entropy_thresh[itr]
        should_halt = entropy < thresh

        # patience
        if self._patience_counter is None or self._patience_counter.size(0) != bs:
            self._patience_counter = torch.zeros(bs, device=device)
        self._patience_counter = torch.where(
            should_halt,
            self._patience_counter + 1,
            torch.zeros_like(self._patience_counter),
        )
        halt_prob = (self._patience_counter >= self.patience).float()
        return halt_prob

    def reset_patience(self, bs, device):
        self._patience_counter = torch.zeros(bs, device=device)

    def _binary_entropy(self, probs):
        eps = 1e-7
        probs = probs.clamp(eps, 1 - eps)
        return -(probs * probs.log() + (1 - probs) * (1 - probs).log())

    def _attention_entropy(self, attn_weights, mask=None):
        eps = 1e-8
        ent = -(attn_weights * (attn_weights + eps).log()).sum(-1)
        if mask is not None:
            m = mask.squeeze(1) if mask.ndim == 3 else mask
            ent = (ent * m.unsqueeze(1)).sum(-1) / m.sum(-1, keepdim=True).clamp(min=1)
        else:
            ent = ent.mean(-1)
        return ent.mean(-1)

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0, requires_grad=False)

@register_halting("pondernet")
class PonderNetHalting(nn.Module):
    def __init__(self, d_model, max_iters=8, prior_p=0.5, kl_weight=0.01):
        super().__init__()
        self.max_iters = max_iters
        self.prior_p = prior_p
        self.kl_weight = kl_weight

        self.halt_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

        self._halt_lambdas = []

    def _pool(self, hidden, mask=None):
        if hidden.dim() == 3:
            if mask is not None:
                m = mask.squeeze(1) if mask.ndim == 3 else mask
                m_float = m.unsqueeze(1).float()
                return (hidden * m_float).sum(-1) / m_float.sum(-1).clamp(min=1)
            return hidden.mean(-1)
        return hidden

    def forward(self, hidden, mask=None, itr=0, attn_weights=None, sample_fn=None):
        h = self._pool(hidden, mask)
        lambda_n = torch.sigmoid(self.halt_head(h).squeeze(-1))
        return lambda_n

    def forward_train(self, hidden, mask, itr):
        h = self._pool(hidden, mask)
        lambda_n = torch.sigmoid(self.halt_head(h).squeeze(-1))
        self._halt_lambdas.append(lambda_n)
        return lambda_n

    def compute_halting_distribution(self):
        if not self._halt_lambdas:
            return None

        eps = 1e-7
        lambdas = torch.stack(self._halt_lambdas)
        lambdas = lambdas.clamp(eps, 1 - eps)

        halt_dist = []
        log_survival = torch.zeros_like(lambdas[0])

        for i in range(len(lambdas)):
            if i < len(lambdas) - 1:
                p_n = torch.exp(log_survival) * lambdas[i]
                halt_dist.append(p_n)
                log_survival = log_survival + torch.log(1 - lambdas[i])
            else:
                halt_dist.append(torch.exp(log_survival))

        return torch.stack(halt_dist)

    def compute_kl_loss(self):
        halt_dist = self.compute_halting_distribution()
        if halt_dist is None:
            return torch.tensor(0.0)

        n_iters = halt_dist.size(0)
        device = halt_dist.device

        n = torch.arange(1, n_iters + 1, device=device).float()
        log_prior = (n - 1) * math.log(1 - self.prior_p) + math.log(self.prior_p)
        log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
        log_prior = log_prior[:, None]

        log_halt = torch.log(halt_dist + 1e-7)
        kl = (halt_dist * (log_halt - log_prior)).sum(0).mean()

        return self.kl_weight * kl

    def reset(self):
        self._halt_lambdas = []