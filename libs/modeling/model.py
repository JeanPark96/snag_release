import torch
import torch.nn as nn

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
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
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