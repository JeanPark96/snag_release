from copy import deepcopy
from gc import enable

import torch.nn as nn
import torch
from .blocks import LayerNorm, TransformerDecoder, LoopedTransformerDecoder, MaskedConv1D, MINDFusionBlock
from .halt_methods import build_halting

modules = dict()
def register_fusion(name):
    def decorator(module):
        modules[name] = module
        return module
    return decorator


@register_fusion('xattn')
class XAttNFusion(nn.Module):
    """ Fuse video and text features using attention.
    """

    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # text feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (adaln | affine)
        return_features=True,  # whether to return intermediate features for analysis
    ):
        super(XAttNFusion, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)

        self.apply(self.__init_weights__)

        self._last_attn_weights = []

        self.return_features = return_features

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        features = {} if self.return_features else None
        for idx, layer in enumerate(self.layers):
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
            if self.return_features:
                features[f"fusion_layer_{idx+1}_q"] = q
                features[f"fusion_layer_{idx+1}_q_mask"] = q_mask
            self._last_attn_weights.append(layer._last_attn_weights)
        q = self.ln_out(q)

        if self.return_features:
                features["fusion_ln_out_q"] = q
                features["fusion_ln_out_q_mask"] = q_mask

        # repeat query to match the size of key / value
        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)
            if self.return_features:
                features["fusion_repeated_q"] = q
                features["fusion_repeated_q_mask"] = q_mask

        return q, q_mask, features

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)
        
        all_features = {} if self.return_features else None
        idx = 0
        out, out_masks = tuple(), tuple()
        for x, mask in zip(vid, vid_masks):
            x, mask, features = self._forward(x, mask, text, text_mask, text_size)
            out += (x, )
            out_masks += (mask, )
            idx+= 1
            if self.return_features:
                all_features[f"fusion_pyramid_{idx}"] = features

        return out, out_masks, all_features


@register_fusion('xattn_level')
class XAttNFusion(nn.Module):
    def __init__(
        self,
        vid_dim,
        text_dim,
        n_layers=2,
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        return_features=True,
        branch_depths=None,  # e.g. [2, 2, 3, 3, 4, 4]
    ):
        super(XAttNFusion, self).__init__()

        self.branch_depths = branch_depths
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)
        self.apply(self.__init_weights__)
        self._last_attn_weights = []
        self.return_features = return_features

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None, depth=None):
        if depth is None:
            depth = self.n_layers
        
        features = {} if self.return_features else None
        for idx, layer in enumerate(self.layers[:depth]):
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
            if self.return_features:
                features[f"fusion_layer_{idx+1}_q"] = q
                features[f"fusion_layer_{idx+1}_q_mask"] = q_mask
            self._last_attn_weights.append(layer._last_attn_weights)
        q = self.ln_out(q)

        if self.return_features:
            features["fusion_ln_out_q"] = q
            features["fusion_ln_out_q_mask"] = q_mask

        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)
            if self.return_features:
                features["fusion_repeated_q"] = q
                features["fusion_repeated_q_mask"] = q_mask

        return q, q_mask, features

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            depth = self.branch_depths[0] if self.branch_depths is not None else None
            return self._forward(vid, vid_masks, text, text_mask, text_size, depth=depth)

        all_features = {} if self.return_features else None
        out, out_masks = tuple(), tuple()
        for idx, (x, mask) in enumerate(zip(vid, vid_masks)):
            depth = self.branch_depths[idx] if self.branch_depths is not None else None
            x, mask, features = self._forward(x, mask, text, text_mask, text_size, depth=depth)
            out += (x,)
            out_masks += (mask,)
            if self.return_features:
                all_features[f"fusion_pyramid_{idx+1}"] = features

        return out, out_masks, all_features
    
@register_fusion('looped_xattn')
class LoopedXAttNFusion(nn.Module):
    """Fuse video and text features using looped cross-attention.
    
    Instead of N separate decoder layers, one shared decoder is
    iterated N times — each iteration re-attends to the query text,
    progressively refining the video–query alignment.
    """

    def __init__(
        self,
        vid_dim,
        text_dim,
        n_layers=2,             # now means number of loop iterations
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        conv_kernel=3,          # ConvSwiGLU kernel
        use_loop_embed=False,
    ):
        super().__init__()

        self.n_iterations = n_layers  # reinterpret as loop count

        # ONE shared decoder block (weight-tied across iterations)
        self.decoder = LoopedTransformerDecoder(
            vid_dim, text_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            xattn_mode=xattn_mode,
            conv_kernel=conv_kernel,
            use_loop_embed=use_loop_embed,
            max_loops=n_layers,
        )

        self.ln_out = LayerNorm(vid_dim)
        self.apply(self.__init_weights__)
        self._last_attn_weights = []

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None, level_idx=0):
        self._last_attn_weights = []

        # iteratively re-attend to text, refining alignment each time
        if level_idx == 3:
            for i in range(self.n_iterations):
                    q, q_mask = self.decoder(
                        q, q_mask, kv, kv_mask, kv_size, loop_idx=i
                    )
                    self._last_attn_weights.append(
                        self.decoder._last_attn_weights
                    )
        else:
                # for later FPN levels, just run through the shared decoder once (no loop)
                q, q_mask = self.decoder(
                    q, q_mask, kv, kv_mask, kv_size, loop_idx=0
                )
                self._last_attn_weights.append(
                    self.decoder._last_attn_weights
                )

        q = self.ln_out(q)

        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)

        out, out_masks = tuple(), tuple()
        level_idx = 0
        for x, mask in zip(vid, vid_masks):
            level_idx += 1
            x, mask = self._forward(x, mask, text, text_mask, text_size, level_idx)
            out += (x,)
            out_masks += (mask,)

        return out, out_masks

@register_fusion('looped_xattn_act')
class LoopedXAttNFusionACT(nn.Module):
    """Fuse video and text features using looped cross-attention.
    
    Instead of N separate decoder layers, one shared decoder is
    iterated N times — each iteration re-attends to the query text,
    progressively refining the video–query alignment.
    """

    def __init__(
        self,
        vid_dim,
        text_dim,
        n_layers=2,             # now means number of loop iterations
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        conv_kernel=3,          # ConvSwiGLU kernel
        use_loop_embed=False,
        decoder_halting='entropy',    # halting method for looped decoder (None for no halting)
        warmup_depth=3
    ):
        super().__init__()

        self.halting = build_halting(decoder_halting)
        self.warmup_depth = warmup_depth

        self.n_iterations = n_layers  # reinterpret as loop count

        # ONE shared decoder block (weight-tied across iterations)
        self.decoder = LoopedTransformerDecoder(
            vid_dim, text_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            xattn_mode=xattn_mode,
            conv_kernel=conv_kernel,
            use_loop_embed=use_loop_embed,
            max_loops=n_layers,
        )

        self.ln_out = LayerNorm(vid_dim)
        self.apply(self.__init_weights__)
        self._last_attn_weights = []

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _set_decoder_dropout(self, enable):
        """Toggle dropout layers in decoder for clean eval vs MC sampling."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Dropout):
                if enable:
                    m.train()
                else:
                    m.eval()

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        self._last_attn_weights = []
        bs = q.size(0)
        device = q.device

        halt_at_eval_only = getattr(self, 'halt_at_eval_only', False)
        use_halting_loop = (self.halting is not None) and (not self.training or not halt_at_eval_only)

        if not use_halting_loop:
            # original path — full depth during training, or no halting module
            for i in range(self.n_iterations):
                q, q_mask = self.decoder(q, q_mask, kv, kv_mask, kv_size, loop_idx=i)
                self._last_attn_weights.append(self.decoder._last_attn_weights)

                # collect variance stats for eval-time halting
                if self.training and self.halting is not None and hasattr(self.halting, 'forward_warmup'):
                    sample_fn = lambda: self.decoder(
                        q, q_mask, kv, kv_mask, kv_size, loop_idx=i
                    )[0]
                    self.halting.forward_warmup(
                        hidden=q, mask=q_mask, itr=i, sample_fn=sample_fn
                    )

            q = self.ln_out(q)
            if kv_size is not None and q.size(0) != kv.size(0):
                q = q.repeat_interleave(kv_size, dim=0)
                q_mask = q_mask.repeat_interleave(kv_size, dim=0)
            return q, q_mask

        # --- adaptive path (eval, or training with halt_at_eval_only=False) ---
        halting_active = getattr(self, 'halting_active', True)

        halted = torch.zeros(bs, dtype=torch.bool, device=device)
        accumulated = torch.zeros_like(q)
        iters_used = torch.zeros(bs, dtype=torch.long, device=device)
        all_halt_probs = []

        for i in range(self.n_iterations):
            active = ~halted
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]

            q_active = q[active_idx]
            q_mask_active = q_mask[active_idx]
            kv_active = kv[active_idx]
            kv_mask_active = kv_mask[active_idx]
            kv_size_active = kv_size[active_idx] if kv_size is not None else None
            
            # MAIN forward pass — dropout OFF for clean representations
            self._set_decoder_dropout(False)

            q_active, q_mask_active = self.decoder(
                q_active, q_mask_active,
                kv_active, kv_mask_active,
                kv_size_active, loop_idx=i
            )

            self._set_decoder_dropout(True)  # restore for sampling


            self._last_attn_weights.append(
                self.decoder._last_attn_weights
            )

            q = q.clone()
            q[active_idx] = q_active

            iters_used[active_idx] += 1

            #print(f"[DEBUG] iter {i}, halting_active={halting_active}, "
          #f"n_active={active_idx.size(0)}, training={self.training}")

            if not halting_active:
                if hasattr(self.halting, 'forward_warmup'):
                    sample_fn = lambda: self.decoder(
                        q_active, q_mask_active,
                        kv_active, kv_mask_active,
                        kv_size_active, loop_idx=i
                    )[0]
                    self.halting.forward_warmup(
                        hidden=q_active,
                        mask=q_mask_active,
                        itr=i,
                        sample_fn=sample_fn,
                        attn_weights=self.decoder._last_attn_weights,
                    )
                newly_halted = (iters_used[active_idx] >= self.warmup_depth)

            else:
                sample_fn = lambda: self.decoder(
                    q_active, q_mask_active,
                    kv_active, kv_mask_active,
                    kv_size_active, loop_idx=i
                )[0]
                halt_prob = self.halting(
                    hidden=q_active,
                    mask=q_mask_active,
                    itr=i,
                    attn_weights=self.decoder._last_attn_weights,
                    sample_fn=sample_fn,
                )

                # print directly, no hasattr guard
                # print(f"[Decoder] iter {i}: "
                #     f"halt_prob={halt_prob.item():.4f}, "
                #     f"var_iter0={self.halting.var_iter0.item():.4f}, "
                #     f"thresh={self.halting.threshold * self.halting.var_iter0.item():.4f}")

                # # also check if _last_var exists
                # print(f"  has _last_var: {hasattr(self.halting, '_last_var')}")
                # if hasattr(self.halting, '_last_var'):
                #     print(f"  var={self.halting._last_var.mean():.4f}")

                if i == 0:
                    n_halt_iter0 = (halt_prob > 0.5).sum().item()
                    n_total = halt_prob.size(0)
                    print(f"[Decoder split] halt@iter0: {n_halt_iter0}/{n_total} "
                        f"({100*n_halt_iter0/max(n_total,1):.1f}%)")

                    # accumulate for final summary
                    if not hasattr(self, '_halt_iter0_count'):
                        self._halt_iter0_count = 0
                        self._halt_iter0_total = 0
                    self._halt_iter0_count += n_halt_iter0
                    self._halt_iter0_total += n_total

                    # store for IoU analysis
                    self._last_halted_at_iter0 = (halt_prob > 0.5).item()

                full_halt_prob = torch.zeros(bs, device=device)
                full_halt_prob[active_idx] = halt_prob
                all_halt_probs.append(full_halt_prob)
                newly_halted = halt_prob > 0.5

                # variance logging
                if hasattr(self.halting, '_last_var'):
                    if not hasattr(self, '_var_log'):
                        self._var_log = []
                    self._var_log.append({
                        'itr': self.halting._last_itr,
                        'var_mean': self.halting._last_var.mean().item(),
                        'var_std': self.halting._last_var.std().item(),
                        'ratio': self.halting._last_var.mean().item() / self.halting.var_iter0.item(),
                        'thresh': self.halting._last_thresh.item() if torch.is_tensor(self.halting._last_thresh) else self.halting._last_thresh,
                        'n_halted': (halt_prob > 0.5).sum().item(),
                        'n_active': active_idx.size(0),
                    })

            if i == self.n_iterations - 1:
                newly_halted = torch.ones(active_idx.size(0), dtype=torch.bool, device=device)

            halt_indices = active_idx[newly_halted]
            accumulated[halt_indices] = q[halt_indices]
            halted[halt_indices] = True

        q = self.ln_out(accumulated)

        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        var_log = getattr(self, '_var_log', [])
        self._var_log = []

        return q, q_mask, {
            'iters_used': iters_used,
            'halt_probs': all_halt_probs,
            'var_log': var_log,
        }

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)

        out, out_masks = tuple(), tuple()
        all_halt_info = []

        for x, mask in zip(vid, vid_masks):
            result = self._forward(x, mask, text, text_mask, text_size)
            if len(result) == 3:
                x, mask, halt_info = result
                all_halt_info.append(halt_info)
            else:
                x, mask = result
            out += (x,)
            out_masks += (mask,)

        if all_halt_info:
            agg_iters = torch.stack(
                [h['iters_used'] for h in all_halt_info]
            ).float().mean(0)
            agg_probs = [p for h in all_halt_info for p in h['halt_probs']]
            agg_var_log = [entry for h in all_halt_info for entry in h.get('var_log', [])]
            return out, out_masks, {
                'iters_used': agg_iters,
                'halt_probs': agg_probs,
                'var_log': agg_var_log,
            }
        else:
            return out, out_masks

@register_fusion('looped_xattn_ponder')
class LoopedXAttNFusionACT(nn.Module):
    """Fuse video and text features using looped cross-attention.
    
    Instead of N separate decoder layers, one shared decoder is
    iterated N times — each iteration re-attends to the query text,
    progressively refining the video–query alignment.
    """

    def __init__(
        self,
        vid_dim,
        text_dim,
        n_layers=2,             # now means number of loop iterations
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        conv_kernel=3,          # ConvSwiGLU kernel
        use_loop_embed=False,
        decoder_halting='entropy',    # halting method for looped decoder (None for no halting)
        warmup_depth=3
    ):
        super().__init__()

        self.halting = build_halting(decoder_halting)
        self.warmup_depth = warmup_depth

        self.n_iterations = n_layers  # reinterpret as loop count

        # ONE shared decoder block (weight-tied across iterations)
        self.decoder = LoopedTransformerDecoder(
            vid_dim, text_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            xattn_mode=xattn_mode,
            conv_kernel=conv_kernel,
            use_loop_embed=use_loop_embed,
            max_loops=n_layers,
        )

        self.ln_out = LayerNorm(vid_dim)
        self.apply(self.__init_weights__)
        self._last_attn_weights = []

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        self._last_attn_weights = []
        bs = q.size(0)
        device = q.device

        halt_at_eval_only = getattr(self, 'halt_at_eval_only', False)
        use_halting_loop = (self.halting is not None) and (not self.training or not halt_at_eval_only)

        if not use_halting_loop:
            # TRAINING: full depth, collect per-iteration outputs
            per_iter_outputs = []

            for i in range(self.n_iterations):
                q, q_mask = self.decoder(q, q_mask, kv, kv_mask, kv_size, loop_idx=i)
                self._last_attn_weights.append(self.decoder._last_attn_weights)

                if self.training and self.halting is not None:
                    if hasattr(self.halting, 'forward_train'):
                        self.halting.forward_train(hidden=q, mask=q_mask, itr=i)
                        # store this iteration's output for exact PonderNet
                        per_iter_outputs.append((q.clone(), q_mask.clone()))

                    elif hasattr(self.halting, 'forward_warmup'):
                        sample_fn = lambda: self.decoder(
                            q, q_mask, kv, kv_mask, kv_size, loop_idx=i
                        )[0]
                        self.halting.forward_warmup(
                            hidden=q, mask=q_mask, itr=i, sample_fn=sample_fn
                        )

            q = self.ln_out(q)
            if kv_size is not None and q.size(0) != kv.size(0):
                q = q.repeat_interleave(kv_size, dim=0)
                q_mask = q_mask.repeat_interleave(kv_size, dim=0)

            if self.halting is not None and per_iter_outputs:
                per_iter_normed = []
                for q_i, qm_i in per_iter_outputs:
                    q_normed = self.ln_out(q_i)
                    if kv_size is not None and q_normed.size(0) != kv.size(0):
                        q_normed = q_normed.repeat_interleave(kv_size, dim=0)
                        qm_i = qm_i.repeat_interleave(kv_size, dim=0)
                    per_iter_normed.append((q_normed, qm_i))

                # compute halt distribution and KL before reset
                halt_dist = self.halting.compute_halting_distribution()
                kl_loss = self.halting.compute_kl_loss()

                return q, q_mask, {
                    'per_iter_outputs': per_iter_normed,
                    'iters_used': torch.full((bs,), self.n_iterations, device=device, dtype=torch.float),
                    'halt_dist': halt_dist,
                    'kl_loss': kl_loss,
                    'halt_probs': [],
                }

            return q, q_mask

        else:
            # EVAL: adaptive loop with hard halting
            halted = torch.zeros(bs, dtype=torch.bool, device=device)
            accumulated = torch.zeros_like(q)
            iters_used = torch.zeros(bs, dtype=torch.long, device=device)
            all_halt_probs = []

            halting_active = getattr(self, 'halting_active', True)

            for i in range(self.n_iterations):
                active = ~halted
                if not active.any():
                    break

                active_idx = active.nonzero(as_tuple=True)[0]
                q_active = q[active_idx]
                q_mask_active = q_mask[active_idx]
                kv_active = kv[active_idx]
                kv_mask_active = kv_mask[active_idx]
                kv_size_active = kv_size[active_idx] if kv_size is not None else None

                self._set_decoder_dropout(False)
                q_active, q_mask_active = self.decoder(
                    q_active, q_mask_active,
                    kv_active, kv_mask_active,
                    kv_size_active, loop_idx=i
                )
                self._set_decoder_dropout(True)

                self._last_attn_weights.append(self.decoder._last_attn_weights)

                q = q.clone()
                q[active_idx] = q_active
                iters_used[active_idx] += 1

                halt_prob = self.halting(
                    hidden=q_active, mask=q_mask_active, itr=i
                )

                full_halt_prob = torch.zeros(bs, device=device)
                full_halt_prob[active_idx] = halt_prob
                all_halt_probs.append(full_halt_prob)
                newly_halted = torch.bernoulli(halt_prob).bool()

                if i == self.n_iterations - 1:
                    newly_halted = torch.ones(
                        active_idx.size(0), dtype=torch.bool, device=device
                    )

                halt_indices = active_idx[newly_halted]
                accumulated[halt_indices] = q[halt_indices]
                halted[halt_indices] = True

            q = self.ln_out(accumulated)
            if kv_size is not None and q.size(0) != kv.size(0):
                q = q.repeat_interleave(kv_size, dim=0)
                q_mask = q_mask.repeat_interleave(kv_size, dim=0)

            return q, q_mask, {
                'iters_used': iters_used,
                'halt_probs': all_halt_probs,
            }

    def _set_decoder_dropout(self, enable):
        if self.training:
            return
        for m in self.decoder.modules():
            if isinstance(m, nn.Dropout):
                if enable:
                    m.train()
                else:
                    m.eval()

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)

        out, out_masks = tuple(), tuple()
        all_halt_info = []

        for x, mask in zip(vid, vid_masks):
            # reset halting state before each FPN level
            if self.halting is not None and hasattr(self.halting, 'reset'):
                self.halting.reset()

            result = self._forward(x, mask, text, text_mask, text_size)
            if len(result) == 3:
                x, mask, halt_info = result
                all_halt_info.append(halt_info)
            else:
                x, mask = result
            out += (x,)
            out_masks += (mask,)

        if all_halt_info:
            agg_iters = torch.stack(
                [h['iters_used'] for h in all_halt_info]
            ).float().mean(0)
            agg_probs = [p for h in all_halt_info for p in h.get('halt_probs', [])]

            # aggregate halt_dist and intermediates per level
            agg_halt_dist = None
            agg_kl_loss = torch.tensor(0.0, device=out[0].device)
            agg_per_iter_outputs = []

            for h in all_halt_info:
                if 'kl_loss' in h:
                    agg_kl_loss = agg_kl_loss + h['kl_loss']
                if 'halt_dist' in h and h['halt_dist'] is not None:
                    agg_halt_dist = h['halt_dist']
                if 'per_iter_outputs' in h:
                    agg_per_iter_outputs.append(h['per_iter_outputs'])

            # restructure: group by iteration across FPN levels
            # agg_per_iter_outputs[level][iter] = (q, qm)
            # → we want per_iter_fused_fpns[iter] = (fpn_tuple, fpn_masks_tuple)
            per_iter_fused_fpns = None
            if agg_per_iter_outputs:
                n_iters = len(agg_per_iter_outputs[0])
                per_iter_fused_fpns = []
                for itr in range(n_iters):
                    fpn_iter = tuple(agg_per_iter_outputs[level][itr][0] 
                                for level in range(len(agg_per_iter_outputs)))
                    fpn_masks_iter = tuple(agg_per_iter_outputs[level][itr][1]
                                        for level in range(len(agg_per_iter_outputs)))
                    per_iter_fused_fpns.append((fpn_iter, fpn_masks_iter))

            return out, out_masks, {
                'iters_used': agg_iters,
                'halt_probs': agg_probs,
                'halt_dist': agg_halt_dist,
                'kl_loss': agg_kl_loss,
                'per_iter_fused_fpns': per_iter_fused_fpns,
            }
@register_fusion('looped_xattn_inpred')
class LoopedXAttNFusion(nn.Module):
    def __init__(
        self,
        vid_dim,
        text_dim,
        n_iterations=3,
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        conv_kernel=3,
        use_loop_embed=False,
    ):
        super().__init__()
        self.n_iterations = n_iterations

        # single shared decoder (same as before)
        self.decoder = LoopedTransformerDecoder(
            vid_dim, text_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            xattn_mode=xattn_mode,
            conv_kernel=conv_kernel,
            use_loop_embed=use_loop_embed,
            max_loops=n_iterations,
        )

        self.ln_out = LayerNorm(vid_dim)

        # lightweight: prediction (3 channels) → feature residual
        # cls_score(1) + reg_offset(2) = 3
        self.feedback_proj = MaskedConv1D(3, vid_dim, 1)
        # gate so feedback starts near zero
        self.feedback_gate = nn.Parameter(torch.zeros(1))

        self.apply(self.__init_weights__)
        self._last_attn_weights = []

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None,
                 cls_head=None, reg_head=None):
        """
        q:    video features at one FPN level (B, C, T)
        kv:   text features (B, C_t, S)
        cls_head, reg_head: prediction heads (passed in, not owned)
        
        Returns:
            q:       final refined features
            q_mask:  mask
            all_cls: list of cls predictions per iteration
            all_reg: list of reg predictions per iteration
        """
        self._last_attn_weights = []
        all_cls, all_reg = [], []
        q_input = q  # save original video features

        for i in range(self.n_iterations):
            # fuse: video attends to text
            q, q_mask = self.decoder(
                q, q_mask, kv, kv_mask, kv_size, loop_idx=i
            )
            self._last_attn_weights.append(
                self.decoder._last_attn_weights
            )

            # predict at this iteration (if heads are available)
            if cls_head is not None and reg_head is not None:
                # detach so prediction head gradients don't
                # flow back through the feedback path twice
                cls_pred = cls_head(q)
                reg_pred = reg_head(q)
                all_cls.append(cls_pred)
                all_reg.append(reg_pred)

                # inject prediction back for next iteration
                if i < self.n_iterations - 1:
                    pred_signal = torch.cat([
                        cls_pred.detach().sigmoid(),  # (B, 1, T)
                        reg_pred.detach(),            # (B, 2, T)
                    ], dim=1)                         # (B, 3, T)

                    feedback, _ = self.feedback_proj(pred_signal, q_mask)
                    gate = self.feedback_gate.sigmoid()

                    # start from original features + feedback
                    # NOT from current q — avoids drift
                    q = q_input + gate * feedback

                    if kv_size is not None and q.size(0) != q_input.size(0):
                        q_input_expanded = q_input.repeat_interleave(
                            kv_size, dim=0
                        )
                        q = q_input_expanded + gate * feedback

        q = self.ln_out(q)

        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask, all_cls, all_reg

    def forward(self, vid, vid_masks, text, text_mask, text_size=None,
                cls_head=None, reg_head=None):
        if not isinstance(vid, tuple):
            return self._forward(
                vid, vid_masks, text, text_mask, text_size,
                cls_head, reg_head
            )

        out, out_masks = tuple(), tuple()
        all_cls_levels, all_reg_levels = [], []

        for x, mask in zip(vid, vid_masks):
            x, mask, cls_list, reg_list = self._forward(
                x, mask, text, text_mask, text_size,
                cls_head, reg_head
            )
            out += (x,)
            out_masks += (mask,)
            all_cls_levels.append(cls_list)
            all_reg_levels.append(reg_list)

        return out, out_masks, all_cls_levels, all_reg_levels

@register_fusion("mind_xattn")
class MINDXAttNFusion(nn.Module):
    """
    Drop-in replacement for XAttNFusion.

    Identical interface:
        (vid, vid_masks, text, text_mask, text_size) -> (out, out_masks)

    Changed: instead of nn.ModuleList of n unique TransformerDecoder layers,
    uses a single MINDFusionBlock called n_iterations times (weight-tied).
    """
    def __init__(
        self,
        vid_dim,
        text_dim,
        n_layers=2,     # was n_layers in original
        n_heads=4,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        xattn_mode='adaln',
        use_conv_swiglu=False,
        conv_kernel=3,
        refinement_mode="log_attn",
    ):
        super().__init__()

        self.n_iterations = n_layers

        # single weight-tied block (replaces nn.ModuleList of n layers)
        self.block = MINDFusionBlock(
            vid_dim, text_dim,
            n_heads=n_heads,
            stride=1,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            xattn_mode=xattn_mode,
            use_conv_swiglu=use_conv_swiglu,
            conv_kernel=conv_kernel,
            refinement_mode=refinement_mode,
        )

        self.ln_out = LayerNorm(vid_dim)

        self.apply(self.__init_weights__)

        self._last_attn_weights = []

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        self._last_attn_weights = []
        prev_out, prev_attn = None, None

        # expand q upfront so batch dims are consistent across iterations
        if kv_size is not None:
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        for _ in range(self.n_iterations):
            q_prev = q  # safe now, already expanded
            q, q_mask, prev_attn = self.block(
                q, q_mask, kv, kv_mask, kv_size=None,  # already expanded
                prev_out=prev_out, prev_attn=prev_attn
            )
            prev_out = q_prev
            self._last_attn_weights.append(prev_attn)

        q = self.ln_out(q)
        return q, q_mask

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)

        out, out_masks = tuple(), tuple()
        for x, mask in zip(vid, vid_masks):
            x, mask = self._forward(x, mask, text, text_mask, text_size)
            out += (x,)
            out_masks += (mask,)

        return out, out_masks

def make_fusion(opt):
    opt = deepcopy(opt)
    return modules[opt.pop('name')](**opt)