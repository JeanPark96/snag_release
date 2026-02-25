from copy import deepcopy

import torch.nn as nn
import torch
from .blocks import LayerNorm, TransformerDecoder, LoopedTransformerDecoder, MaskedConv1D, MINDFusionBlock


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

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        for layer in self.layers:
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
            self._last_attn_weights.append(layer._last_attn_weights)
        q = self.ln_out(q)

        # repeat query to match the size of key / value
        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)
            
        out, out_masks = tuple(), tuple()
        for x, mask in zip(vid, vid_masks):
            x, mask = self._forward(x, mask, text, text_mask, text_size)
            out += (x, )
            out_masks += (mask, )

        return out, out_masks
    
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

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        self._last_attn_weights = []

        # iteratively re-attend to text, refining alignment each time
       
        for i in range(self.n_iterations):
                q, q_mask = self.decoder(
                    q, q_mask, kv, kv_mask, kv_size, loop_idx=i
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
        for x, mask in zip(vid, vid_masks):
            x, mask = self._forward(x, mask, text, text_mask, text_size)
            out += (x,)
            out_masks += (mask,)

        return out, out_masks

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