from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MINDEncoderBlockWithLatents, sinusoid_encoding, MaskedConv1D, LayerNorm, TransformerEncoder, LoopedTransformerBlock, MINDEncoderBlock, MINDBranchLevel
)


backbones = dict()
def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator


@register_video_net('transformer')
class VideoTransformer(nn.Module):
    """
    A backbone that combines convolutions with transformer encoder layers 
    to build a feature pyramid.
    
    video clip features
    -> [embedding convs x L1]
    -> [stem transformer x L2]
    -> [branch transformer x L3]
    -> latent video feature pyramid
    """
    def __init__(
        self,
        in_dim,             # video feature dimension
        embd_dim,           # embedding dimension
        max_seq_len,        # max sequence length
        n_heads,            # number of attention heads for MHA
        mha_win_size,       # local window size for MHA (0 for global attention)
        stride=1,           # conv stride applied to the input features
        arch=(2, 1, 6),     # (#convs, #stem transformers, #branch transformers)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        use_abs_pe=False,   # whether to apply absolute position encoding
    ):
        super().__init__()

        assert len(arch) == 3, '(embed convs, stem, branch)'
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # embedding convs
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # stem transformers
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerEncoder(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        # branch transformers (for FPN)
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerEncoder(
                    embd_dim,
                    stride=2 if idx > 0 else 1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # position encoding
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # stem layers
        for block in self.stem:
            x, mask = block(x, mask)

        # branch layers
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x, )
            fpn_masks += (mask, )

        return fpn, fpn_masks

@register_video_net('transformer_looped')
class LoopedVideoTransformer(VideoTransformer):
    def __init__(
        self,
        *args,
        loop_num=4,        # fixed loop count (start here)
        use_adaptive_halt=False,  # flip on later
        halt_threshold=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print("get loop number", loop_num)
        self.n_iterations = loop_num
    
        self.use_adaptive_halt = use_adaptive_halt
        
        if use_adaptive_halt:
            # per-position halting score from features
            self.halt_head = nn.Sequential(
                nn.Linear(kwargs.get('embd_dim', args[1]), 1),
                nn.Sigmoid()
            )
            self.halt_threshold = halt_threshold

    def forward(self, x, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        # === embedding (unchanged) ===
        x, _ = self.embd_fc(x, mask)
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # === position encoding (unchanged) ===
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # === LOOPED STEM ===
        if not self.use_adaptive_halt:
            # Phase 1: fixed iterations — weight-shared
            for i in range(self.n_iterations):
                for block in self.stem:
                    x, mask = block(x, mask)
        else:
            # Phase 2: adaptive halting (ACT-style)
            bs, c, t = x.size()
            halted = torch.zeros(bs, 1, t, device=x.device, dtype=torch.bool)
            cumul_halt = torch.zeros(bs, 1, t, device=x.device)
            running_x = x.clone()
            
            for i in range(self.n_iterations):  # n_iterations = max cap
                for block in self.stem:
                    running_x, mask = block(running_x, mask)
                
                # (bs, c, t) -> (bs, t, c) -> halt score -> (bs, 1, t)
                h = self.halt_head(running_x.transpose(1, 2)).transpose(1, 2)
                cumul_halt = cumul_halt + h
                
                # positions that just crossed threshold
                newly_halted = (cumul_halt >= 1.0) & ~halted
                
                # update only non-halted positions
                update_mask = (~halted).float()
                x = x * (1 - update_mask) + running_x * update_mask
                
                halted = halted | newly_halted
                
                if halted.all():
                    break

        # === branch layers (unchanged) ===
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x,)
            fpn_masks += (mask,)

        return fpn, fpn_masks

@register_video_net('all_looped_transformer')
class LoopedVideoTransformer(VideoTransformer):
    def __init__(
        self,
        *args,
        embd_dim,
        n_heads,
        mha_win_size,
        use_loop_embed,
        attn_pdrop,
        proj_pdrop,
        path_pdrop,
        loop_num=4,        # fixed loop count (start here)
        use_adaptive_halt=False,  # flip on later
        halt_threshold=0.01,
        **kwargs,
    ):
        # pass shared args to parent explicitly
        super().__init__(
            *args,
            embd_dim=embd_dim,
            n_heads=n_heads,
            mha_win_size=mha_win_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            **kwargs,
        )
        print("get loop number", loop_num)
        self.n_iterations = loop_num
    
        self.use_adaptive_halt = use_adaptive_halt
        self.stem = nn.ModuleList()
        
        self.stem.append(
            LoopedTransformerBlock(
                embd_dim, n_heads=n_heads,
                window_size=mha_win_size,
                conv_kernel=3,
                max_loops=loop_num,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_loop_embed=use_loop_embed,
            )
        )
    
        
        if use_adaptive_halt:
            # per-position halting score from features
            self.halt_head = nn.Sequential(
                nn.Linear(kwargs.get('embd_dim', args[1]), 1),
                nn.Sigmoid()
            )
            self.halt_threshold = halt_threshold

    def forward(self, x, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        # === embedding (unchanged) ===
        x, _ = self.embd_fc(x, mask)
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # === position encoding (unchanged) ===
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # === LOOPED STEM ===
        if self.use_adaptive_halt:
            # Phase 1: fixed iterations — weight-shared
            for i in range(self.n_iterations):
                for block in self.stem:
                    x, mask = block(x, mask)
        else:
            # looped stem (THIS IS THE CHANGE)
            for i in range(self.n_iterations):
                for block in self.stem:
                    x, mask = block(x, mask, loop_idx=i)

        # === branch layers (unchanged) ===
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x,)
            fpn_masks += (mask,)

        return fpn, fpn_masks
    
@register_video_net('mind_transformer')
class MINDVideoTransformer(nn.Module):
    """
    Drop-in replacement for VideoTransformer.

    Identical structure:
        video features
        -> [embedding convs x L1]       (SAME)
        -> [stem: MIND looped block]     (CHANGED: weight-tied with refinement)
        -> [branch transformer x L3]     (SAME: regular TransformerEncoder for FPN)
        -> latent video feature pyramid

    arch = (n_embd_convs, n_stem_iterations, n_branch_transformers)
    NOTE: arch[1] now means number of MIND iterations, not number of stem layers.
    """
    def __init__(
        self,
        in_dim,
        embd_dim,
        max_seq_len,
        n_heads,
        mha_win_size,
        stride=1,
        arch=(2, 4, 6),     # (embed convs, MIND iterations, branch transformers)
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_conv_swiglu=False,
        conv_kernel=3,
    ):
        super().__init__()

        assert len(arch) == 3
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.n_stem_iterations = arch[1]

        # === SAME AS ORIGINAL: embedding projection ===
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # === SAME AS ORIGINAL: embedding convs ===
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # === SAME AS ORIGINAL: position encoding ===
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # === CHANGED: stem is now a single MIND block looped N times ===
        self.stem = MINDEncoderBlock(
            embd_dim,
            n_heads=n_heads,
            stride=1,  # stem is always stride=1
            window_size=mha_win_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_conv_swiglu=use_conv_swiglu,
            conv_kernel=conv_kernel,
        )

        # === SAME AS ORIGINAL: branch transformers for FPN ===
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerEncoder(
                    embd_dim,
                    stride=2 if idx > 0 else 1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        # === SAME: embedding projection ===
        x, _ = self.embd_fc(x, mask)

        # === SAME: embedding convs ===
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # === SAME: position encoding ===
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # === CHANGED: MIND looped stem ===
        prev_attn = None
        for _ in range(self.n_stem_iterations):
            x, mask, prev_attn = self.stem(x, mask, prev_attn)

        # === SAME: branch layers building FPN ===
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x,)
            fpn_masks += (mask,)

        return fpn, fpn_masks

@register_video_net('mind_transformer_v2')
class MINDVideoTransformerV2(nn.Module):
    """
    Drop-in replacement for VideoTransformer.
    MIND looping on BOTH stem and each branch level.

    arch = (n_embd_convs, n_stem_iterations, n_branch_levels)

    Args:
        branch_iterations: number of MIND iterations per branch level.
            Can be int (same for all levels) or list (per level).
    """
    def __init__(
        self,
        in_dim,
        embd_dim,
        max_seq_len,
        n_heads,
        mha_win_size,
        stride=1,
        arch=(2, 4, 6),
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_conv_swiglu=False,
        conv_kernel=3,
        branch_iterations=2,  # MIND iterations per branch level
    ):
        super().__init__()

        assert len(arch) == 3
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.n_stem_iterations = arch[1]
        n_branch_levels = arch[2]

        # normalize branch_iterations to a list
        if isinstance(branch_iterations, int):
            self.branch_iterations = [branch_iterations] * n_branch_levels
        else:
            assert len(branch_iterations) == n_branch_levels
            self.branch_iterations = list(branch_iterations)

        # === SAME: embedding projection ===
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # === SAME: embedding convs ===
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # === SAME: position encoding ===
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # === MIND stem: single block looped N times ===
        self.stem = MINDEncoderBlock(
            embd_dim,
            n_heads=n_heads,
            stride=1,
            window_size=mha_win_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_conv_swiglu=use_conv_swiglu,
            conv_kernel=conv_kernel,
        )

        # === MIND branch: each level has its own looped block ===
        self.branch = nn.ModuleList()
        for idx in range(n_branch_levels):
            self.branch.append(
                MINDBranchLevel(
                    embd_dim,
                    n_iterations=self.branch_iterations[idx],
                    n_heads=n_heads,
                    stride=2 if idx > 0 else 1,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_conv_swiglu=use_conv_swiglu,
                    conv_kernel=conv_kernel,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # position encoding
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # MIND stem loop
        prev_attn = None
        for _ in range(self.n_stem_iterations):
            x, mask, prev_attn = self.stem(x, mask, prev_attn)

        # MIND branch levels (each loops independently)
        fpn, fpn_masks = tuple(), tuple()
        for branch_level in self.branch:
            x, mask = branch_level(x, mask)
            fpn += (x,)
            fpn_masks += (mask,)

        return fpn, fpn_masks

@register_video_net('mind_transformer_latent')
class MINDVideoTransformer(nn.Module):
    """
    Drop-in replacement for VideoTransformer.

    Identical structure:
        video features
        -> [embedding convs x L1]       (SAME)
        -> [stem: MIND looped block]     (CHANGED: weight-tied with refinement)
        -> [branch transformer x L3]     (SAME: regular TransformerEncoder for FPN)
        -> latent video feature pyramid

    arch = (n_embd_convs, n_stem_iterations, n_branch_transformers)
    NOTE: arch[1] now means number of MIND iterations, not number of stem layers.
    """
    def __init__(
        self,
        in_dim,
        embd_dim,
        max_seq_len,
        n_heads,
        mha_win_size,
        stride=1,
        arch=(2, 4, 6),     # (embed convs, MIND iterations, branch transformers)
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_conv_swiglu=False,
        conv_kernel=3,
        n_latent=8
    ):
        super().__init__()

        assert len(arch) == 3
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.n_stem_iterations = arch[1]

        # === SAME AS ORIGINAL: embedding projection ===
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # === SAME AS ORIGINAL: embedding convs ===
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # === SAME AS ORIGINAL: position encoding ===
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # === CHANGED: stem is now a single MIND block looped N times ===
        self.stem = MINDEncoderBlockWithLatents(
            embd_dim,
            n_latent=n_latent,          # new param, add to config
            n_heads=n_heads,
            stride=1,
            window_size=mha_win_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_conv_swiglu=use_conv_swiglu,
            conv_kernel=conv_kernel,
        )

        # === SAME AS ORIGINAL: branch transformers for FPN ===
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerEncoder(
                    embd_dim,
                    stride=2 if idx > 0 else 1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)

        # === SAME: embedding projection ===
        x, _ = self.embd_fc(x, mask)

        # === SAME: embedding convs ===
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # === SAME: position encoding ===
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # === CHANGED: MIND looped stem ===
        latent = None  # block initializes from self.latent_tokens on first call
        prev_attn = None
        for _ in range(self.n_stem_iterations):
            x, mask, latent, prev_attn = self.stem(x, mask, latent, prev_attn)

        # === SAME: branch layers building FPN ===
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x,)
            fpn_masks += (mask,)

        return fpn, fpn_masks

def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)