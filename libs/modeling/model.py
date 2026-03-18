from .halt_methods import build_halting
import torch
import torch.nn as nn
import torch.nn.functional as F
from .level_gating import LevelGatingNetwork

from .blocks import MaskedConv1D
from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net


class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])

        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        self._last_fused_fpn = fpn          # <-- ADD THIS LINE

        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)

        

        # ---- diagnostic: per-level prediction analysis ----
        # if not self.training:
        #     print("=" * 80)
        #     for l in range(len(fpn_logits)):
        #         logits_l = fpn_logits[l]     # (B, T_l)
        #         offsets_l = fpn_offsets[l]   # (B, T_l, 2)
        #         mask_l = fpn_masks[l]        # (B, T_l)
        #         feat_l = fpn[l]             # (B, C, T_l)

        #         valid = mask_l.bool()        # (B, T_l)

        #         # cls probs
        #         probs = logits_l.sigmoid()
        #         vp = probs[valid]

        #         # reg offsets — select valid positions then look at the 2 channels
        #         vo = offsets_l[valid]        # (N_valid, 2)

        #         # feat norms
        #         fn = feat_l.norm(dim=1)      # (B, T_l)
        #         vn = fn[valid]

        #         print(f"  Level {l}: T={logits_l.shape[1]:4d} | "
        #             f"cls  mean={vp.mean():.4f} std={vp.std():.4f} "
        #             f"max={vp.max():.4f} min={vp.min():.4f} | "
        #             f"reg  mean={vo.abs().mean():.4f} std={vo.std():.4f} | "
        #             f"feat_norm  mean={vn.mean():.2f} std={vn.std():.2f}")
        #     print("=" * 80)

   
        return fpn_logits, fpn_offsets, fpn_masks

    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_offsets, fpn_masks


class PtTransformerGate(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])

        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

        self.hard_coded = False

        # --- level gating (Method B) ---
        gate_opt = opt.get('level_gate', None)
        if gate_opt is not None and gate_opt.get('enabled', False):
            self.level_gate = LevelGatingNetwork(
                num_levels=opt['vid_net']['arch'][-1],
                fpn_dim=opt['vid_net']['embd_dim'],
                text_dim=opt['text_net']['embd_dim'],
                hidden_dim=gate_opt.get('hidden_dim', 128),
                init_bias=gate_opt.get('init_bias', 2.0),
                tau=gate_opt.get('tau_init', 2.0),
                tau_min=gate_opt.get('tau_min', 0.5),
            )
        else:
            self.level_gate = None

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        
        if self.hard_coded:
            active_levels = [2, 3, 4]  # hardcoded for validation
        
            # Only fuse active levels
            active_fpn = tuple(fpn[l] for l in active_levels)
            active_masks = tuple(fpn_masks[l] for l in active_levels)
            
            fused, fused_masks = self.fusion(
                active_fpn, active_masks, text, text_masks, text_size
            )
            self._last_fused_fpn = fpn 
            fused_logits, _ = self.cls_head(fused, fused_masks)
            fused_offsets, fused_out_masks = self.reg_head(fused, fused_masks)
            
            # Use the actual (expanded) batch size from fused outputs
            bs = fused_logits[0].size(0)
            device = fused_logits[0].device

            all_logits, all_offsets, all_masks = tuple(), tuple(), tuple()
            active_idx = 0
            for l in range(len(fpn)):
                if l in active_levels:
                    all_logits += (fused_logits[active_idx],)
                    all_offsets += (fused_offsets[active_idx],)
                    all_masks += (fused_out_masks[active_idx],)
                    active_idx += 1
                else:
                    # p = number of temporal positions at this FPN level
                    # fpn_masks[l] is (bs_original, 1, p) — get p from it
                    p = fpn_masks[l].size(-1)
                    all_logits += (torch.full((bs, p), -1e4, device=device),)
                    all_offsets += (torch.zeros(bs, p, 2, device=device),)
                    all_masks += (torch.zeros(bs, p, dtype=torch.bool, device=device),)

            return all_logits, all_offsets, all_masks
        else:
            if self.level_gate is not None:
                gates = self.level_gate(
                    fpn, fpn_masks, text, text_masks, text_size
                )
            else:
                gates = None
    
            # --- fuse all levels (needed for differentiability during training) ---
            fused_fpn, fused_masks = self.fusion(
                fpn, fpn_masks, text, text_masks, text_size
            )
            self._last_fused_fpn = fpn 
            # --- apply soft gates to fused features ---
            if gates is not None:
                gated_fpn = tuple()
                for l, f in enumerate(fused_fpn):
                    g = gates[:, l].unsqueeze(1).unsqueeze(2)   # (bs, 1, 1)
                    gated_fpn += (f * g,)
            else:
                gated_fpn = fused_fpn
    
            # --- predict ---
            fpn_logits, _ = self.cls_head(gated_fpn, fused_masks)
            fpn_offsets, fpn_masks_out = self.reg_head(gated_fpn, fused_masks)
    
            # detached so gradient flows only through feature scaling, not this mask
            # --- mask logits for gated-out levels (INFERENCE ONLY) ---
            if gates is not None and not self.training:
                masked_logits = tuple()
                for l, logits_l in enumerate(fpn_logits):
                    penalty = (1.0 - gates[:, l]).unsqueeze(1) * (-1e4)
                    masked_logits += (logits_l + penalty,)
                fpn_logits = masked_logits
    
            return fpn_logits, fpn_offsets, fpn_masks_out, gates

    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks, gates = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_offsets, fpn_masks, gates

class CrossLevelContext(nn.Module):
    """
    Learns to select which coarse level provides the best
    spatial prior for fine levels, then projects that context
    into a form fine levels can use.
    """
    def __init__(self, embd_dim, n_source_levels=4, n_target_levels=3):
        super().__init__()
        # soft attention over source levels
        # input: pooled feature per source level → weight per source
        self.source_gate = nn.Sequential(
            nn.Linear(embd_dim * n_source_levels, n_source_levels),
            nn.Softmax(dim=-1)
        )
        # project cross-level context into target feature space
        self.context_proj = nn.Conv1d(embd_dim, embd_dim, 1)

        # initialize near zero so iteration 0 = baseline
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, fused_fpn, fpn_masks):
        B, C = fused_fpn[0].shape[0], fused_fpn[0].shape[1]

        # Step 1: per-source-level summary
        source_summaries = []
        for l in [2, 3, 4, 5]:
            feat = fused_fpn[l]                              # (B, 256, T_l)
            mask = fpn_masks[l].float()                       # (B, 1, T_l) — already has channel dim
            pooled = (feat * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-8)  # (B, 256)
            source_summaries.append(pooled)

        # Step 2: soft attention to select source
        gate_input = torch.cat(source_summaries, dim=-1)      # (B, 1024)
        gate_weights = self.source_gate(gate_input)            # (B, 4)

        # Step 3: weighted combination upsampled to finest resolution
        T_finest = fused_fpn[0].shape[2]
        combined_context = torch.zeros(B, C, T_finest,
                                    device=fused_fpn[0].device)
        for i, l in enumerate([2, 3, 4, 5]):
            upsampled = F.interpolate(
                fused_fpn[l].detach(), size=T_finest,
                mode='nearest'
            )
            combined_context += gate_weights[:, i:i+1, None] * upsampled

        # Step 4: project
        projected = self.context_proj(combined_context)

        # Step 5: inject into fine levels
        enriched = list(fused_fpn)
        for l in [0, 1, 2]:
            T_l = fused_fpn[l].shape[2]
            if T_l != T_finest:
                ctx_l = F.interpolate(projected, size=T_l,
                                    mode='nearest')
            else:
                ctx_l = projected
            enriched[l] = fused_fpn[l] + ctx_l

        return tuple(enriched)

class PtTransformerACT(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])

        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        out = self.vid_net(vid, vid_masks)
        if isinstance(out, tuple) and len(out) == 3:
            fpn, fpn_masks, halt_info = out
        else:
            fpn, fpn_masks = out
            halt_info = None
        return fpn, fpn_masks, halt_info

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        out = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        if isinstance(out, tuple) and len(out) == 3:
            fpn, fpn_masks, halt_info = out
        else:
            fpn, fpn_masks = out
            halt_info = None

        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks, halt_info

    def forward(self, vid, vid_masks, text, text_masks, text_size=None):

        halt_at_eval_only = getattr(self, 'halt_at_eval_only', False)
        self.vid_net.halt_at_eval_only = halt_at_eval_only
        self.fusion.halt_at_eval_only = halt_at_eval_only
        
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat([t[:k] for t, k in zip(text_masks, text_size)])

        text, text_masks = self.encode_text(text, text_masks)

        # propagate halting flags
        halting_active = getattr(self, 'halting_active', True)
        fixed_warmup_depth = getattr(self, 'fixed_warmup_depth', 3)
        self.vid_net.halting_active = halting_active
        self.vid_net.fixed_warmup_depth = fixed_warmup_depth
        self.fusion.halting_active = halting_active
        self.fusion.fixed_warmup_depth = fixed_warmup_depth

        # propagate probe_fn BEFORE encode_video and fuse_and_predict
        for module in [self.vid_net, self.fusion]:
            halting = getattr(module, 'halting', None)
            if halting is not None and hasattr(halting, 'probe_fn'):
                # CRITICAL: use object.__setattr__ to avoid nn.Module
                # registering cls_head as a submodule of halting,
                # which would duplicate parameters and break EMA
                object.__setattr__(halting, 'probe_fn', self.cls_head)

        fpn, fpn_masks, enc_halt_info = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks, dec_halt_info = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        if not self.training:
            return fpn_logits, fpn_offsets, fpn_masks

        # --- aggregate halting info for trainer ---
        device = vid.device
        bs = vid.size(0)
        halt_loss = torch.tensor(0.0, device=device)
        iters_used = {}

        if enc_halt_info is not None:
            iters_used['encoder'] = enc_halt_info['iters_used']
            enc_halting = getattr(self.vid_net, 'halting', None)
            if enc_halting is not None and enc_halt_info['halt_probs']:
                halt_loss = halt_loss + enc_halting.compute_loss(
                    enc_halt_info['halt_probs'],
                    enc_halt_info['iters_used'],
                    self.vid_net.n_stem_iterations,
                )
        else:
            iters_used['encoder'] = torch.full(
            (bs,), self.vid_net.n_stem_iterations, device=device, dtype=torch.float
        )

            

        if dec_halt_info is not None:
            iters_used['decoder'] = dec_halt_info['iters_used']
            dec_halting = getattr(self.fusion, 'halting', None)
            if dec_halting is not None and dec_halt_info['halt_probs']:
                halt_loss = halt_loss + dec_halting.compute_loss(
                    dec_halt_info['halt_probs'],
                    dec_halt_info['iters_used'],
                    self.fusion.n_iterations,
                )
        else:
            iters_used['decoder'] = torch.full(
            (bs,), self.fusion.n_iterations, device=device, dtype=torch.float
        )

        halt_info_dict = {
            'halt_loss': halt_loss,
            'iters_used': iters_used,
        }

        if enc_halt_info is not None and 'var_log' in enc_halt_info:
            halt_info_dict['enc_var_log'] = enc_halt_info['var_log']
        
        if dec_halt_info is not None and 'var_log' in dec_halt_info:
            halt_info_dict['dec_var_log'] = dec_halt_info['var_log']

        return fpn_logits, fpn_offsets, fpn_masks, halt_info_dict

class PtTransformerACTPonder(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])

        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        out = self.vid_net(vid, vid_masks)
        if isinstance(out, tuple) and len(out) == 3:
            fpn, fpn_masks, halt_info = out
        else:
            fpn, fpn_masks = out
            halt_info = None
        return fpn, fpn_masks, halt_info

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        out = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        if isinstance(out, tuple) and len(out) == 3:
            fpn, fpn_masks, halt_info = out
        else:
            fpn, fpn_masks = out
            halt_info = None

        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks, halt_info

    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat([t[:k] for t, k in zip(text_masks, text_size)])

        # propagate flags
        halting_active = getattr(self, 'halting_active', True)
        halt_at_eval_only = getattr(self, 'halt_at_eval_only', False)
        for attr in ['halting_active', 'fixed_warmup_depth', 'halt_at_eval_only']:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self.vid_net, attr, val)
                setattr(self.fusion, attr, val)

        # propagate probe_fn
        for module in [self.vid_net, self.fusion]:
            halting = getattr(module, 'halting', None)
            if halting is not None and hasattr(halting, 'probe_fn'):
                object.__setattr__(halting, 'probe_fn', self.cls_head)

        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks, enc_halt_info = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks, dec_halt_info = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        if not self.training:
            return fpn_logits, fpn_offsets, fpn_masks

        bs = vid.size(0)
        device = vid.device
        halt_loss = torch.tensor(0.0, device=device)
        iters_used = {}

        enc_intermediate_preds = None
        dec_intermediate_preds = None

        # --- encoder ---
        if enc_halt_info is not None:
            iters_used['encoder'] = enc_halt_info.get(
                'iters_used',
                torch.full((bs,), self.vid_net.n_stem_iterations, device=device, dtype=torch.float)
            )

            if 'intermediate_fpns' in enc_halt_info and 'halt_dist' in enc_halt_info:
                enc_intermediate_preds = []
                for fpn_i, fpn_masks_i in enc_halt_info['intermediate_fpns']:
                    fused_i, fused_masks_i = self.fusion(
                        fpn_i, fpn_masks_i, text, text_masks, text_size
                    )
                    if isinstance(fused_i, tuple) and len(fused_i) == 3:
                        fused_i, fused_masks_i, _ = fused_i
                    logits_i, _ = self.cls_head(fused_i, fused_masks_i)
                    offsets_i, masks_i = self.reg_head(fused_i, fused_masks_i)
                    enc_intermediate_preds.append((logits_i, offsets_i, masks_i))

            # pick up kl_loss or halt_loss
            if 'kl_loss' in enc_halt_info:
                halt_loss = halt_loss + enc_halt_info['kl_loss']
            elif 'halt_loss' in enc_halt_info:
                halt_loss = halt_loss + enc_halt_info['halt_loss']
        else:
            iters_used['encoder'] = torch.full(
                (bs,), self.vid_net.n_stem_iterations, device=device, dtype=torch.float
            )

        # --- decoder ---
        if dec_halt_info is not None:
            iters_used['decoder'] = dec_halt_info.get(
                'iters_used',
                torch.full((bs,), self.fusion.n_iterations, device=device, dtype=torch.float)
            )

            # decoder returns 'per_iter_outputs', not 'intermediate_fused_fpns'
            if 'per_iter_fused_fpns' in dec_halt_info and 'halt_dist' in dec_halt_info:
                dec_intermediate_preds = []
                for fpn_i, fpn_masks_i in dec_halt_info['per_iter_fused_fpns']:
                    logits_i, _ = self.cls_head(fpn_i, fpn_masks_i)
                    offsets_i, masks_i = self.reg_head(fpn_i, fpn_masks_i)
                    dec_intermediate_preds.append((logits_i, offsets_i, masks_i))


            # pick up kl_loss or halt_loss
            if 'kl_loss' in dec_halt_info:
                halt_loss = halt_loss + dec_halt_info['kl_loss']
            elif 'halt_loss' in dec_halt_info:
                halt_loss = halt_loss + dec_halt_info['halt_loss']
        else:
            iters_used['decoder'] = torch.full(
                (bs,), self.fusion.n_iterations, device=device, dtype=torch.float
            )

        return fpn_logits, fpn_offsets, fpn_masks, {
            'halt_loss': halt_loss,
            'iters_used': iters_used,
            'enc_intermediate_preds': enc_intermediate_preds,
            'enc_halt_dist': enc_halt_info.get('halt_dist') if enc_halt_info else None,
            'dec_intermediate_preds': dec_intermediate_preds,
            'dec_halt_dist': dec_halt_info.get('halt_dist') if dec_halt_info else None,
        }
        
class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())
    
class PtTransformerLoop(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

        # NEW: number of refinement iterations
        # print(opt.keys())
        self.n_refine = opt.get('n_iterations', 1)
        print(self.n_refine, " iterations")

        # NEW: project prediction (3ch) → vid_dim, one per iteration
        if self.n_refine > 1:
            vid_dim = opt['vid_net']['embd_dim']
            self.feedback_projs = nn.ModuleList([
                MaskedConv1D(3, vid_dim, 1)
                for _ in range(self.n_refine - 1)
            ])
            # initialize near zero so iteration 0 matches baseline
            for proj in self.feedback_projs:
                nn.init.zeros_(proj.conv.weight)
                if proj.conv.bias is not None:
                    nn.init.zeros_(proj.conv.bias)

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn_orig = fpn
        fpn_masks_orig = fpn_masks  # <-- save original masks too
        all_logits, all_offsets = [], []

        for i in range(self.n_refine):
            # always feed original-shaped masks into fusion
            fused, fused_masks = self.fusion(
                fpn, fpn_masks_orig, text, text_masks, text_size
            )

            fpn_logits, _ = self.cls_head(fused, fused_masks)
            fpn_offsets, fpn_masks_out = self.reg_head(fused, fused_masks)

            all_logits.append(fpn_logits)
            all_offsets.append(fpn_offsets)

            if i < self.n_refine - 1:
                new_fpn = []
                for l in range(len(fpn_orig)):
                    # add this right before the sig = torch.cat line
                    # print(f"fpn_logits type: {type(fpn_logits)}, len: {len(fpn_logits)}")
                    # print(f"fpn_logits[0] shape: {fpn_logits[0].shape}")
                    # print(f"fpn_offsets[0] shape: {fpn_offsets[0].shape}")
                    # print(f"fused_masks type: {type(fused_masks)}, fused_masks[0] shape: {fused_masks[0].shape}")
                    sig = torch.cat([
                        fpn_logits[l].detach().sigmoid().unsqueeze(1),
                        fpn_offsets[l].detach().permute(0, 2, 1),
                    ], dim=1)

                    # use expanded masks for the projection
                    fb, _ = self.feedback_projs[i](
                        sig, fused_masks[l]
                    )

                    # aggregate back to original batch if expanded
                    if fb.size(0) != fpn_orig[l].size(0):
                        splits = text_size.tolist()
                        fb = torch.stack([
                            s.mean(dim=0)
                            for s in torch.split(fb, splits, dim=0)
                        ])

                    new_fpn.append(fpn_orig[l] + fb)

                fpn = tuple(new_fpn)

        return fpn_logits, fpn_offsets, fpn_masks_out, all_logits, all_offsets

    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )

        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)

        result = self.fuse_and_predict(
            fpn, fpn_masks, text, text_masks, text_size
        )

        # unpack: final predictions + intermediate lists
        fpn_logits, fpn_offsets, fpn_masks, all_logits, all_offsets = result

        return fpn_logits, fpn_offsets, fpn_masks, all_logits, all_offsets

class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points