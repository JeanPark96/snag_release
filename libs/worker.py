from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import os
import shutil
import time

import numpy as np
from .modeling.halt_methods import MCDropoutHalting
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, PtTransformer, PtTransformerGate,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str
from .modeling import PtTransformerACTPonder as PtTransformerACT

import numpy as np

class ACTTrainer:

    def __init__(self, opt):

        self.opt = opt

        self.halt_at_eval_only = opt['train'].get('halt_at_eval_only', False)

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        self.model = PtTransformerACT(opt['model']).cuda()
        for i, (name, p) in enumerate(self.model.named_parameters()):
            if i in [302, 303, 304, 305]:
                print(f"Param {i}: {name}, shape={p.shape}")
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()])
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        # adaptive computation hyperparameters
        self.halt_warmup_epochs = opt['train'].get('halt_warmup_epochs', 5)
        self.halt_weight = opt['train'].get('halt_weight', 0.01)
        self.max_iters = opt['train'].get('max_iters', 6)
        self.halt_stages = opt['train'].get('halt_stages', ['encoder'])
        self.fixed_warmup_depth = opt['train'].get('fixed_warmup_depth', 3)

        model = self._unwrap(self.model) if hasattr(self, '_unwrap') else self.model
        self.vid_net_iters = getattr(model.vid_net, 'n_stem_iterations', 4)
        self.fusion_iters = getattr(model.fusion, 'n_iterations', 2)

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
            
                if self.clip_grad_norm:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                    if get_rank() == 0:
                        if 'grad_norm' not in self.loss_meters:
                            self.loss_meters['grad_norm'] = AverageMeter()
                        self.loss_meters['grad_norm'].update(grad_norm.detach())
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        """
        loss_norm update.
        
        The only change: we accumulate the new diagnostic keys and pass them 
        through. The loss_norm EMA itself is fine — it operates on the 
        accumulated `norm` which is computed on final (post-halt) outputs.
        The key insight is that norm must come from the ACCUMULATED outputs,
        not from any intermediate iteration. Since we compute norm after the
        loop in _microbatch_forward_backward, this is already correct.
        """
        cls_loss = reg_loss = halt_loss = total_loss = norm = 0
        enc_iters = dec_iters = frac_maxed = 0

        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            halt_loss += loss_dict['halt']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            enc_iters += loss_dict['enc_iters']
            dec_iters += loss_dict['dec_iters']
            frac_maxed += loss_dict['frac_maxed']

        # loss_norm EMA update (unchanged logic, operates on final outputs)
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )

        n_mb = self.num_microbatches
        return {
            'cls': cls_loss,
            'reg': reg_loss,
            'halt': halt_loss,
            'total': total_loss,
            # average over microbatches for logging
            'enc_iters': enc_iters / n_mb,
            'dec_iters': dec_iters / n_mb,
            'frac_maxed': frac_maxed / n_mb,
        }


    # def _microbatch_forward_backward(self, data_list, is_last=False):
    #     """
    #     Adaptive computation version.
        
    #     Key changes from original:
    #     - Iterative forward loop with per-sample halting
    #     - Gradient scaling by 1/iters_used per sample
    #     - DDP sync wraps the entire loop + backward, not individual iters
    #     - Returns halting diagnostics alongside losses
    #     """
    #     # ---- batch data (unchanged) ----
    #     vid, vid_masks, text, text_masks, text_size = self._batchify(
    #         vid_list=[d['vid'] for d in data_list],
    #         text_list=[d['text'] for d in data_list]
    #     )
    #     vid = vid.cuda(non_blocking=True)
    #     vid_masks = vid_masks.cuda(non_blocking=True)
    #     text = text.cuda(non_blocking=True)
    #     text_masks = text_masks.cuda(non_blocking=True)
    #     text_size = text_size.cuda(non_blocking=True)

    #     targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
    #     targets = targets.cuda(non_blocking=True)
    #     bs = vid.size(0)

    #     # ---- determine whether halting is active or in warmup ----
    #     model = self._unwrap(self.model)
    #     halting_active = self.epoch >= self.halt_warmup_epochs  # local copy
    #     model.halting_active = halting_active                    # propagate to model
    #     model.fixed_warmup_depth = self.fixed_warmup_depth
    #     model.halt_at_eval_only = self.halt_at_eval_only  # NEW


    #     # ---- DDP sync context: wrap the ENTIRE loop + backward ----
    #     # This is the fix for sync timing. We only sync gradients on the 
    #     # last microbatch, and the context wraps everything including backward().
    #     if is_last or not self.opt['_distributed']:
    #         ctx = nullcontext()
    #     else:
    #         ctx = self.model.no_sync()

    #     # with ctx:
    #     #     out = self.model(vid, vid_masks, text, text_masks, text_size)
    #     #     fpn_logits, fpn_offsets, fpn_masks, halt_info = out
    #     #     iters_used = halt_info['iters_used']
    #     #     halt_loss = halt_info['halt_loss']

    #     #     if get_rank() == 0 and self.itr % self.log_interval == 0:
    #     #         enc_var_log = halt_info.get('enc_var_log', [])
    #     #         if enc_var_log:
    #     #             log_str = f"\n  [Encoder MC Dropout Variance]"
    #     #             log_str += f"\n  var_iter0: {self._unwrap(self.model).vid_net.halting.var_iter0.item():.6f}"
    #     #             for entry in enc_var_log:
    #     #                 log_str += (
    #     #                     f"\n  iter {entry['itr']}: "
    #     #                     f"var={entry['var_mean']:.6f} ± {entry['var_std']:.6f}, "
    #     #                     f"thresh={entry['thresh']:.6f}, "
    #     #                     f"halted={entry['n_halted']}/{entry['n_active']}"
    #     #                 )
    #     #             print0(log_str)
    #     #         dec_var_log = halt_info.get('dec_var_log', [])
    #     #         if dec_var_log:
    #     #             log_str = f"\n  [Decoder MC Dropout Variance]"
    #     #             log_str += f"\n  var_iter0: {self._unwrap(self.model).fusion.halting.var_iter0.item():.6f}"
    #     #             for entry in dec_var_log:
    #     #                 log_str += (
    #     #                     f"\n  iter {entry['itr']}: "
    #     #                     f"var={entry['var_mean']:.6f} ± {entry['var_std']:.6f}, "
    #     #                     f"thresh={entry['thresh']:.6f}, "
    #     #                     f"halted={entry['n_halted']}/{entry['n_active']}"
    #     #                 )
    #     #             print0(log_str)

    #     #     fpn_n_points = [m.size(-1) for m in fpn_masks]
    #     #     fpn_points = self.pt_gen(fpn_n_points)

    #     #     # ---- stitch outputs ----
    #     #     fpn_logits = torch.cat(fpn_logits, dim=1)
    #     #     fpn_offsets = torch.cat(fpn_offsets, dim=1)
    #     #     fpn_masks = torch.cat(fpn_masks, dim=1)
    #     #     points = torch.cat(fpn_points)

    #     #     # ---- expand iters_used to match post-fusion batch size ----
    #     #     # fpn_logits is (sum(text_size), p) not (bs, p)
    #     #     fused_bs = fpn_logits.size(0)
    #     #     if fused_bs != bs:
    #     #         # multiple text queries per video — expand per-video iters
    #     #         # to per-query by repeating according to text_size
    #     #         expanded_iters = {}
    #     #         for stage, iters in iters_used.items():
    #     #             if iters.size(0) == bs and fused_bs != bs:
    #     #                 expanded_iters[stage] = iters.repeat_interleave(
    #     #                     text_size.to(iters.device)
    #     #                 )
    #     #             else:
    #     #                 expanded_iters[stage] = iters
    #     #         iters_used = expanded_iters

    #     #     # ---- annotate points ----
    #     #     gt_labels, gt_offsets = self._annotate_points(points, targets)

    #     #     # ---- gradient scaling now uses correct batch size ----
    #     #     pos_masks = torch.logical_and(gt_labels, fpn_masks)
    #     #     norm = pos_masks.sum()

    #     #     total_iters = sum(
    #     #         iters_used[stage] for stage in self.halt_stages
    #     #     ).float()
    #     #     sample_weights = (1.0 / total_iters)
    #     #     sample_weights = sample_weights * (fused_bs / sample_weights.sum())
    #     #     sample_weights = sample_weights.detach()

    #     #     point_weights = sample_weights[:, None].expand_as(fpn_logits)

    #     #     ## (2) classification loss — weighted
    #     #     valid_logits = fpn_logits[fpn_masks]
    #     #     valid_labels = gt_labels[fpn_masks]
    #     #     valid_weights = point_weights[fpn_masks]
    #     #     cls_loss = self._calc_weighted_focal_loss(
    #     #         logits=valid_logits,
    #     #         labels=valid_labels,
    #     #         weights=valid_weights,
    #     #     ) / self.loss_norm * get_world_size()

    #     #     ## (3) regression loss — weighted
    #     #     pos_weights = point_weights[pos_masks]
    #     #     reg_loss = self._calc_weighted_iou_loss(
    #     #         pred_offsets=fpn_offsets[pos_masks],
    #     #         gt_offsets=gt_offsets[pos_masks],
    #     #         weights=pos_weights,
    #     #     ) / self.loss_norm * get_world_size()

    #     #     ## (4) total loss with halting regularization
    #     #     total_loss = cls_loss + self.loss_weight * reg_loss
    #     #     if halting_active and self.halt_weight > 0:
    #     #         total_loss = total_loss + self.halt_weight * halt_loss

    #     #     total_loss.backward()

    #     with ctx:
    #         out = self.model(vid, vid_masks, text, text_masks, text_size)
    #         fpn_logits, fpn_offsets, fpn_masks, halt_info = out
    #         iters_used = halt_info['iters_used']
    #         halt_loss = halt_info['halt_loss']

    #         fpn_n_points = [m.size(-1) for m in fpn_masks]
    #         fpn_points = self.pt_gen(fpn_n_points)
    #         points = torch.cat(fpn_points)

    #         # annotate once (same targets for all iterations)
    #         fpn_logits_cat = torch.cat(fpn_logits, dim=1)
    #         fpn_offsets_cat = torch.cat(fpn_offsets, dim=1)
    #         fpn_masks_cat = torch.cat(fpn_masks, dim=1)
    #         gt_labels, gt_offsets = self._annotate_points(points, targets)
    #         pos_masks = torch.logical_and(gt_labels, fpn_masks_cat)
    #         norm = pos_masks.sum()

    #         # --- exact PonderNet: weighted task loss across iterations ---
    #         enc_intermediate = halt_info.get('enc_intermediate_preds')
    #         enc_halt_dist = halt_info.get('enc_halt_dist')
    #         dec_intermediate = halt_info.get('dec_intermediate_preds')
    #         dec_halt_dist = halt_info.get('dec_halt_dist')

    #         has_pondernet = (enc_intermediate is not None) or (dec_intermediate is not None)

    #         if has_pondernet:
    #             total_task_loss = torch.tensor(0.0, device=vid.device)

    #             # pick whichever stage has intermediates
    #             # if both have them, use decoder (closer to final prediction)
    #             if dec_intermediate is not None and dec_halt_dist is not None:
    #                 intermediates = dec_intermediate
    #                 halt_dist = dec_halt_dist
    #             elif enc_intermediate is not None and enc_halt_dist is not None:
    #                 intermediates = enc_intermediate
    #                 halt_dist = enc_halt_dist

    #             # L_task = Σ p_n * L(y_n, y*)
    #             for n, (logits_n, offsets_n, masks_n) in enumerate(intermediates):
    #                 logits_n = torch.cat(logits_n, dim=1)
    #                 offsets_n = torch.cat(offsets_n, dim=1)
    #                 masks_n = torch.cat(masks_n, dim=1)

    #                 # expand halt_dist to match post-fusion batch size
    #                 p_n = halt_dist[n]  # (bs,)
    #                 if p_n.size(0) != logits_n.size(0):
    #                     p_n = p_n.repeat_interleave(text_size.to(p_n.device))

    #                 # per-sample task loss at this iteration
    #                 pos_masks_n = torch.logical_and(gt_labels, masks_n)
    #                 cls_loss_n = self._calc_focal_loss(
    #                     logits=logits_n[masks_n],
    #                     labels=gt_labels[masks_n],
    #                 ) / self.loss_norm * get_world_size()

    #                 reg_loss_n = self._calc_iou_loss(
    #                     pred_offsets=offsets_n[pos_masks_n],
    #                     gt_offsets=gt_offsets[pos_masks_n],
    #                 ) / self.loss_norm * get_world_size()

    #                 task_loss_n = cls_loss_n + self.loss_weight * reg_loss_n

    #                 # weight by halting probability
    #                 # p_n is per-sample, task_loss_n is scalar (summed over batch)
    #                 # for exact weighting: compute per-sample loss, multiply, sum
    #                 # simplified: weight the batch loss by mean p_n
    #                 total_task_loss = total_task_loss + p_n.mean() * task_loss_n

    #             cls_loss = total_task_loss  # for logging
    #             reg_loss = torch.tensor(0.0, device=vid.device)  # already included
    #             total_loss = total_task_loss + self.halt_weight * halt_loss

    #         else:
    #             # standard loss (no PonderNet intermediates)
    #             cls_loss = self._calc_focal_loss(
    #                 logits=fpn_logits_cat[fpn_masks_cat],
    #                 labels=gt_labels[fpn_masks_cat],
    #             ) / self.loss_norm * get_world_size()

    #             reg_loss = self._calc_iou_loss(
    #                 pred_offsets=fpn_offsets_cat[pos_masks],
    #                 gt_offsets=gt_offsets[pos_masks],
    #             ) / self.loss_norm * get_world_size()

    #             total_loss = cls_loss + self.loss_weight * reg_loss
    #             if self.halt_weight > 0:
    #                 total_loss = total_loss + self.halt_weight * halt_loss

    #         total_loss.backward()

    #     return {
    #         'cls': cls_loss.detach(),
    #         'reg': reg_loss.detach(),
    #         'halt': halt_loss.detach() if torch.is_tensor(halt_loss) else torch.tensor(0.0),
    #         'total': total_loss.detach(),
    #         'norm': norm.detach(),
    #         #'enc_iters': iters_used.get('encoder', torch.ones(fused_bs)).float().mean().detach(),
    #         #'dec_iters': iters_used.get('decoder', torch.ones(fused_bs)).float().mean().detach(),
    #         'frac_maxed': (
    #             sum(iters_used[s] for s in self.halt_stages).float() ==
    #             self.max_iters * len(self.halt_stages)
    #         ).float().mean().detach(),
    #     }

    def _microbatch_forward_backward(self, data_list, is_last=False):
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list],
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        bs = vid.size(0)

        # propagate flags
        model = self._unwrap(self.model)
        
        halting_active = self.epoch >= self.halt_warmup_epochs
        model.halting_active = halting_active
        model.halt_at_eval_only = self.halt_at_eval_only
        model.fixed_warmup_depth = self.fixed_warmup_depth

        if is_last or not self.opt['_distributed']:
            ctx = nullcontext()
        else:
            ctx = self.model.no_sync()

        with ctx:
            out = self.model(vid, vid_masks, text, text_masks, text_size)
            fpn_logits, fpn_offsets, fpn_masks, halt_info = out
            iters_used = halt_info['iters_used']
            halt_loss = halt_info['halt_loss']

            # print(f"halt_loss: {halt_loss.item():.6f}, requires_grad={halt_loss.requires_grad}")

            # enc_halt_dist = halt_info.get('enc_halt_dist')
            # dec_halt_dist = halt_info.get('dec_halt_dist')
            # print(f"enc_halt_dist: {enc_halt_dist is not None}, "
            #     f"requires_grad={enc_halt_dist.requires_grad if enc_halt_dist is not None else 'N/A'}")
            # print(f"dec_halt_dist: {dec_halt_dist is not None}, "
            #     f"requires_grad={dec_halt_dist.requires_grad if dec_halt_dist is not None else 'N/A'}")

            # enc_intermediate = halt_info.get('enc_intermediate_preds')
            # dec_intermediate = halt_info.get('dec_intermediate_preds')
            # print(f"enc_intermediate: {enc_intermediate is not None}, "
            #     f"n={len(enc_intermediate) if enc_intermediate else 0}")
            # print(f"dec_intermediate: {dec_intermediate is not None}, "
            #     f"n={len(dec_intermediate) if dec_intermediate else 0}")

            # model_unwrapped = self._unwrap(self.model)
            # for name, module in [('encoder', model_unwrapped.vid_net), 
            #                     ('decoder', model_unwrapped.fusion)]:
            #     halting = getattr(module, 'halting', None)
            #     if halting is not None:
            #         lambdas = getattr(halting, '_halt_lambdas', [])
            #         print(f"{name}._halt_lambdas: len={len(lambdas)}")
            #         if lambdas:
            #             print(f"  [0].requires_grad={lambdas[0].requires_grad}, "
            #                 f"shape={lambdas[0].shape}")

            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits = torch.cat(fpn_logits, dim=1)
            fpn_offsets = torch.cat(fpn_offsets, dim=1)
            fpn_masks = torch.cat(fpn_masks, dim=1)
            points = torch.cat(fpn_points)

            gt_labels, gt_offsets = self._annotate_points(points, targets)
            pos_masks = torch.logical_and(gt_labels, fpn_masks)
            norm = pos_masks.sum()

            # --- main task loss (on final/combined output) ---
            cls_loss = self._calc_focal_loss(
                logits=fpn_logits[fpn_masks],
                labels=gt_labels[fpn_masks],
            ) / self.loss_norm * get_world_size()

            reg_loss = self._calc_iou_loss(
                pred_offsets=fpn_offsets[pos_masks],
                gt_offsets=gt_offsets[pos_masks],
            ) / self.loss_norm * get_world_size()

            # --- PonderNet weighted intermediate task loss ---
            ponder_loss = torch.tensor(0.0, device=vid.device)

            # encoder intermediates
            enc_intermediate = halt_info.get('enc_intermediate_preds')
            enc_halt_dist = halt_info.get('enc_halt_dist')
            if enc_intermediate is not None and enc_halt_dist is not None:
                for i, (logits_i, offsets_i, masks_i) in enumerate(enc_intermediate):
                    logits_i = torch.cat(logits_i, dim=1)
                    offsets_i = torch.cat(offsets_i, dim=1)
                    masks_i = torch.cat(masks_i, dim=1)

                    # expand to match fused batch size if needed
                    fused_bs = logits_i.size(0)

                    gt_labels_i, gt_offsets_i = self._annotate_points(points, targets)
                    pos_masks_i = torch.logical_and(gt_labels_i, masks_i)

                    cls_i = self._calc_focal_loss(
                        logits=logits_i[masks_i],
                        labels=gt_labels_i[masks_i],
                    ) / self.loss_norm * get_world_size()

                    reg_i = self._calc_iou_loss(
                        pred_offsets=offsets_i[pos_masks_i],
                        gt_offsets=gt_offsets_i[pos_masks_i],
                    ) / self.loss_norm * get_world_size()

                    task_loss_i = cls_i + self.loss_weight * reg_i
                    # weight by halting distribution
                    # enc_halt_dist: (n_iters, bs) → use mean over batch
                    weight_i = enc_halt_dist[i].mean()
                    ponder_loss = ponder_loss + weight_i * task_loss_i

            # decoder intermediates
            dec_intermediate = halt_info.get('dec_intermediate_preds')
            dec_halt_dist = halt_info.get('dec_halt_dist')
            if dec_intermediate is not None and dec_halt_dist is not None:
                for i, (logits_i, offsets_i, masks_i) in enumerate(dec_intermediate):
                    logits_i = torch.cat(logits_i, dim=1)
                    offsets_i = torch.cat(offsets_i, dim=1)
                    masks_i = torch.cat(masks_i, dim=1)

                    fused_bs = logits_i.size(0)

                    gt_labels_i, gt_offsets_i = self._annotate_points(points, targets)
                    pos_masks_i = torch.logical_and(gt_labels_i, masks_i)

                    cls_i = self._calc_focal_loss(
                        logits=logits_i[masks_i],
                        labels=gt_labels_i[masks_i],
                    ) / self.loss_norm * get_world_size()

                    reg_i = self._calc_iou_loss(
                        pred_offsets=offsets_i[pos_masks_i],
                        gt_offsets=gt_offsets_i[pos_masks_i],
                    ) / self.loss_norm * get_world_size()

                    task_loss_i = cls_i + self.loss_weight * reg_i
                    weight_i = dec_halt_dist[i].mean()
                    ponder_loss = ponder_loss + weight_i * task_loss_i
                    print(f"halt_loss(kl): {halt_loss.item():.6f}, "
                        f"ponder_loss: {ponder_loss.item():.6f}, "
                        f"ponder requires_grad: {ponder_loss.requires_grad}")
                    print(f"dec_intermediate: {halt_info.get('dec_intermediate_preds') is not None}")
                    print(f"dec_halt_dist: {halt_info.get('dec_halt_dist') is not None}")

            # --- total loss ---
            total_loss = cls_loss + self.loss_weight * reg_loss
            if self.halt_weight > 0:
                total_loss = total_loss + self.halt_weight * (halt_loss + ponder_loss)

            total_loss.backward()

        fused_bs = fpn_logits.size(0)
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'halt': (halt_loss + ponder_loss).detach() if torch.is_tensor(halt_loss) else torch.tensor(0.0),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'enc_iters': iters_used.get(
                'encoder', torch.full((fused_bs,), self.vid_net_iters, device=vid.device)
            ).float().mean().detach(),
            'dec_iters': iters_used.get(
                'decoder', torch.full((fused_bs,), self.fusion_iters, device=vid.device)
            ).float().mean().detach(),
            'frac_maxed': torch.tensor(0.0),
        }


    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _calc_weighted_focal_loss(self, logits, labels, weights, smoothing=0.2, alpha=0.5):
        """Focal loss with per-point sample weights."""
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        # sigmoid_focal_loss with reduction='none' then manually weight
        per_point_loss = sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='none')
        return (per_point_loss * weights).sum()

    def _calc_weighted_iou_loss(self, pred_offsets, gt_offsets, weights):
        """IoU loss with per-point sample weights."""
        iou_loss_fn = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        per_point_loss = iou_loss_fn(pred_offsets, gt_offsets, reduction='none')
        return (per_point_loss * weights).sum()

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    #@torch.no_grad()
    # def _ema_update(self):
    #     for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
    #         p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))
    
    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))
        # sync buffers (halting stats, running norms, etc.)
        for b, b_ema in zip(self._unwrap(self.model).buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

class ACTEvaluator:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformerACT(opt['model']).cuda()
        self.load_model()

        halting = getattr(self.model.fusion, 'halting', None)
        if halting is not None:
            if hasattr(halting, 'var_iter0'):
                v = halting.var_iter0
                if v.dim() == 0:
                    print0(f"Decoder var_iter0: {v.item():.6f}")
                    print0(f"Decoder initialized: {halting.initialized.item()}")
                else:
                    print0(f"Decoder var_iter0: {v[0].item():.6f}")
            
            #print(f"iter {i}: var_mean={pred_var.mean():.4f}, var_std={pred_var.std():.4f}, var_iter0={self.halting.var_iter0.item():.4f}")
        
        for module in [self.model.vid_net, self.model.fusion]:
            halting = getattr(module, 'halting', None)
            if halting is not None and hasattr(halting, 'probe_fn'):
                object.__setattr__(halting, 'probe_fn', self.model.cls_head)

        self.model.eval().requires_grad_(False)
        # in evaluator, after self.model.eval():
        #print("Setting dropout layers.")
        # force dropout on for MC dropout halting (both encoder and decoder)
        #self._enable_mc_dropout()   
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

        # NEW: halting diagnostics accumulators
        self.halt_stats = {
            'enc_iters': [],
            'dec_iters': [],
        }

    def _enable_mc_dropout(self):
        """
        MC dropout halting requires dropout to be active at eval time.
        Otherwise all T forward passes produce identical outputs
        and variance is always zero.
        
        Only affects modules whose halting type is mc_dropout.
        Other halting methods (entropy, prediction_entropy, calibrated)
        don't need dropout at eval time.
        """
        modules_to_check = [
            ('vid_net', self.model.vid_net),
            ('fusion', self.model.fusion),
        ]

        for name, module in modules_to_check:
            halting = getattr(module, 'halting', None)
            if halting is None:
                continue
            if not isinstance(halting, MCDropoutHalting):
                continue

            # determine which submodule contains the dropout layers
            # encoder: the stem is what gets called in sample_fn
            # decoder: the decoder block is what gets called in sample_fn
            if name == 'vid_net':
                target = getattr(module, 'stem', module)
            elif name == 'fusion':
                target = getattr(module, 'decoder', module)
            else:
                target = module

            count = 0
            for m in target.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
                    count += 1

            print0(f"MC dropout halting: forced {count} dropout layers "
                f"to train mode in {name}.{target.__class__.__name__}")

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print0("Evaluation started.")
        start_time = time.time()
        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']
            assert len(results) == len(targets)

            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                # track IoU by halting depth
                if not hasattr(self, '_iou_by_depth'):
                    self._iou_by_depth = {1: [], 2: []}
                halted_early = getattr(self.model.fusion, '_last_halted_at_iter0', False)
                depth = 1 if halted_early else 2
                self._iou_by_depth[depth].append(iou_topk[0].item() if len(iou_topk) > 0 else 0.0)
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        
        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    def predict(self, data):
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            # CHANGED: unpack optional halt_info from encode_video
            fpn, fpn_masks, enc_halt_info = self.model.encode_video(
                window, window_mask
            )
            if enc_halt_info is not None:
                self.halt_stats['enc_iters'].append(
                    enc_halt_info['iters_used'].cpu()
                )

            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                # CHANGED: unpack optional halt_info from fuse_and_predict
                fpn_logits, fpn_offsets, _, dec_halt_info = \
                    self.model.fuse_and_predict(
                        fpn, fpn_masks, text, text_mask
                    )
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
                if dec_halt_info is not None:
                    self.halt_stats['dec_iters'].append(
                        dec_halt_info['iters_used'].cpu()
                    )

            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            segs, scores = self.batched_nms(segs[idx], scores[idx])

            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _collect_segments(
        self,
        fpn_points, fpn_logits, fpn_offsets, fpn_masks, ext_scores,
    ):
        # completely unchanged
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )

        # NEW: log halting statistics
        if is_last and (self.halt_stats['enc_iters'] or self.halt_stats['dec_iters']):
            log_str += "\n\n--- Halting Statistics ---"
            if self.halt_stats['enc_iters']:
                enc = torch.cat(self.halt_stats['enc_iters']).float()
                log_str += (
                    f"\nEncoder iters: "
                    f"mean={enc.mean():.2f}, "
                    f"std={enc.std():.2f}, "
                    f"min={enc.min():.0f}, "
                    f"max={enc.max():.0f}"
                )
                # distribution histogram
                max_iter = int(enc.max().item())
                for k in range(1, max_iter + 1):
                    pct = (enc == k).float().mean() * 100
                    log_str += f"\n  depth {k}: {pct:.1f}%"

            if self.halt_stats['dec_iters']:
                dec = torch.cat(self.halt_stats['dec_iters']).float()
                log_str += (
                    f"\nDecoder iters: "
                    f"mean={dec.mean():.2f}, "
                    f"std={dec.std():.2f}, "
                    f"min={dec.min():.0f}, "
                    f"max={dec.max():.0f}"
                )
                max_iter = int(dec.max().item())
                for k in range(1, max_iter + 1):
                    pct = (dec == k).float().mean() * 100
                    log_str += f"\n  depth {k}: {pct:.1f}%"
        if is_last:
            fusion = self.model.fusion
            if hasattr(fusion, '_halt_iter0_count'):
                total = fusion._halt_iter0_total
                halted = fusion._halt_iter0_count
                log_str += (
                    f"\n\n[Decoder Summary] halt@iter0: {halted}/{total} "
                    f"({100*halted/max(total,1):.1f}%)"
                )
        if is_last and hasattr(self, '_iou_by_depth'):
            log_str += "\n\n[IoU by decoder depth]"
            for depth, ious in sorted(self._iou_by_depth.items()):
                if ious:
                    mean_iou = sum(ious) / len(ious)
                    log_str += (
                        f"\n  depth {depth}: n={len(ious)}, "
                        f"mean_IoU={mean_iou:.4f}"
                    )

        self.logger.write(log_str)

class Trainer:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        self.model = PtTransformer(opt['model']).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()], find_unused_parameters=False)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks, _ = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks, _ = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }


    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

class TrainerDecGate:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))
        self.use_level_gate = True
        # build model and EMA
        self.model = PtTransformerGate(opt['model']).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()], find_unused_parameters=False)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        # --- level gating (Method B) ---
        gate_opt = opt['model'].get('level_gate', {})
        self.use_level_gate = gate_opt.get('enabled', False)
        self.lambda_compute = gate_opt.get('lambda_compute', 0.5)
        self.tau_init = gate_opt.get('tau_init', 2.0)
        self.tau_min = gate_opt.get('tau_min', 0.5)
        self.gate_warmup_epochs = gate_opt.get('warmup_epochs', 0)

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            # --- gate warmup phase ---
            # model_raw = self._unwrap(self.model)
            # if self.use_level_gate and model_raw.level_gate is not None:
            #     if self.epoch < self.gate_warmup_epochs:
            #         # freeze everything except the gate
            #         for name, p in model_raw.named_parameters():
            #             p.requires_grad = 'level_gate' in name
            #     elif self.epoch == self.gate_warmup_epochs:
            #         # unfreeze everything
            #         for p in model_raw.parameters():
            #             p.requires_grad = True
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            # --- gate warmup: freeze all except level_gate ---
            model_raw = self._unwrap(self.model)
            if (self.use_level_gate
                and model_raw.level_gate is not None):
            
                # tau annealing: linear from tau_init to tau_min over training
                progress = self.epoch / max(self.num_epochs - 1, 1)
                new_tau = self.tau_init + (self.tau_min - self.tau_init) * progress
                model_raw.level_gate.set_tau(new_tau)
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        ret = {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}
        if self.use_level_gate:
            # propagate last microbatch's gate stats for logging
            for k, v in loss_dict.items():
                if k.startswith('g') or k == 'L_compute':
                    ret[k] = v
        return ret
    
    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list],
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)
    
        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
    
        # --- (a) forward pass: unpack 4 return values ---
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks, gates, soft_gates = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks, gates, soft_gates = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
    
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)
    
        # --- (b) build per-point gate weights: (bs, total_points) ---
        if gates is not None:
            gate_weights_list = []
            for l, n_pts in enumerate(fpn_n_points):
                gate_weights_list.append(
                    gates[:, l:l+1].expand(-1, n_pts)
                )
            gate_weights = torch.cat(gate_weights_list, dim=1)
        else:
            gate_weights = None
    
        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1)  # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)      # (bs, p)
        points = torch.cat(fpn_points)               # (p, 4)
    
        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)
    
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()
    
        ## (2) classification loss — weighted by g_l
        # if gate_weights is not None:
        #     # L_task_cls = Σ_l g_l · FocalLoss(c_l, GT_l)
        #     valid_logits = fpn_logits[fpn_masks]
        #     valid_labels = gt_labels[fpn_masks].to(valid_logits.dtype)
        #     valid_labels = valid_labels * 0.8 + 0.1     # smoothing=0.2
        #     valid_weights = gate_weights[fpn_masks]
    
        #     per_point_cls = sigmoid_focal_loss(
        #         valid_logits, valid_labels, alpha=0.5, reduction='none'
        #     )
        #     cls_loss = (per_point_cls * valid_weights).sum() \
        #         / self.loss_norm * get_world_size()
        # else:
        #     cls_loss = self._calc_focal_loss(
        #         logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        #     ) / self.loss_norm * get_world_size()
    
        # ## (3) regression loss — weighted by g_l
        # if gate_weights is not None:
        #     # L_task_reg = Σ_l g_l · DiouLoss(r_l, GT_l)
        #     pos_offsets = fpn_offsets[pos_masks]
        #     pos_gt = gt_offsets[pos_masks]
        #     pos_weights = gate_weights[pos_masks]
    
        #     iou_loss_fn = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        #     if pos_offsets.numel() > 0:
        #         per_point_reg = iou_loss_fn(
        #             pos_offsets, pos_gt, reduction='none'
        #         )
        #         reg_loss = (per_point_reg * pos_weights).sum() \
        #             / self.loss_norm * get_world_size()
        #     else:
        #         reg_loss = 0.0 * fpn_logits.sum()  # keep in graph
        # else:
        #     reg_loss = self._calc_iou_loss(
        #         pred_offsets=fpn_offsets[pos_masks],
        #         gt_offsets=gt_offsets[pos_masks]
        #     ) / self.loss_norm * get_world_size()
        
        # ## (4) compute loss: L_compute = (Σ_l g_l) / L
        # if gates is not None:
        #     compute_loss = self.lambda_compute * gates.mean()
        # else:
        #     compute_loss = 0.0
    
        # ## (5) total loss
        # total_loss = cls_loss + self.loss_weight * reg_loss + compute_loss

        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()

        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks],
            gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        if gates is not None:
            with torch.no_grad():
                scores = torch.sigmoid(fpn_logits) * fpn_masks.float()  # (bs, p)
                
                # decode proposals: center +/- offset * stride
                pt_ctr = points[:, 0]                          # (p,)
                strides = points[:, 3]                          # (p,)
                pred_start = pt_ctr - fpn_offsets[..., 0] * strides  # (bs, p)
                pred_end = pt_ctr + fpn_offsets[..., 1] * strides    # (bs, p)

                # top-5 per query by cls score
                topk = min(5, scores.size(1))
                _, topk_idx = scores.topk(topk, dim=1)         # (bs, 5)

                # gather top-5 segments
                top_start = pred_start.gather(1, topk_idx)     # (bs, 5)
                top_end = pred_end.gather(1, topk_idx)         # (bs, 5)

                # compute IoU of each with GT
                gt_start = targets[:, 0:1]                      # (bs, 1)
                gt_end = targets[:, 1:2]                        # (bs, 1)

                inter_start = torch.max(top_start, gt_start)
                inter_end = torch.min(top_end, gt_end)
                inter = (inter_end - inter_start).clamp(min=0)

                union = (top_end - top_start) + (gt_end - gt_start) - inter
                iou = inter / union.clamp(min=1e-8)            # (bs, 5)

                # reward = best IoU among top-5
                reward = iou.max(dim=1).values                  # (bs,)

            # REINFORCE: reward the gate's selection
            log_prob = (
                gates * torch.log(soft_gates + 1e-8)
                + (1 - gates) * torch.log(1 - soft_gates + 1e-8)
            ).sum(dim=-1)                                       # (bs,)

            gate_loss = -(log_prob * reward).mean()

            total_loss = cls_loss + self.loss_weight * reg_loss \
                    + gate_loss + self.lambda_compute * gates.mean()
        total_loss.backward()
    
        ## (6) build return dict with gate diagnostics
        ret = {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }
        if gates is not None:
            ret['L_compute'] = (self.lambda_compute * gates.mean()).detach()
            gate_mean = gates.detach().mean(0)  # (num_levels,)
            for l in range(gate_mean.size(0)):
                ret['g{}'.format(l)] = gate_mean[l]
        return ret

    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()



class EvaluatorDecGate:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformerGate(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        import numpy as np    # move it here

        print0("Evaluation started.")
        start_time = time.time()
        oracle_stats = []
        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']

            # ---- oracle level analysis ----
            raw = self._all_fpn_raw
            per_level_info = {}
            for q_idx, target in enumerate(targets):
                gt_start, gt_end = target[0], target[1]
                raw = self._all_fpn_raw[q_idx]  # per-query data

                # print(f"\nQuery {q_idx}: GT=({gt_start:.1f}, {gt_end:.1f}), "
                #     f"duration={gt_end - gt_start:.1f}s")

                best_level, best_iou_overall = -1, 0.0
                for level, (points, logits, offsets, masks) in enumerate(
                    zip(raw['fpn_points'], raw['fpn_logits'],
                        raw['fpn_offsets'], raw['fpn_masks'])
                ):
                    lg = logits[0]       # (T_l,)  — already single query
                    off = offsets[0]     # (T_l, 2)
                    mk = masks[0]       # (T_l,)
                    valid = mk.bool()

                    if valid.sum() == 0:
                        print(f"  Level {level}: no valid positions")
                        continue
                    

                    pts = points[valid]       # (N, 2) — center, stride
                    of = off[valid]           # (N, 2)
                    lg_v = lg[valid]          # (N,)

                    # decode to time (same math as _collect_segments)
                    pt_ctr = pts[:, 0]
                    left = pt_ctr - of[:, 0] * pts[:, 1]
                    right = pt_ctr + of[:, 1] * pts[:, 1]

                    # convert feature-space → seconds
                    left_sec = (left * self.vid_stride * raw['clip_stride']
                                + 0.5 * raw['clip_size']) / raw['fps']
                    right_sec = (right * self.vid_stride * raw['clip_stride']
                                + 0.5 * raw['clip_size']) / raw['fps']

                    # IoU with ground truth
                    inter_start = torch.clamp(left_sec, min=gt_start)
                    inter_end = torch.clamp(right_sec, max=gt_end)
                    inter = torch.clamp(inter_end - inter_start, min=0)
                    union = (right_sec - left_sec) + (gt_end - gt_start) - inter
                    iou_vals = inter / (union + 1e-8)

                    best_idx = iou_vals.argmax()
                    best_iou = iou_vals[best_idx].item()
                    best_score = torch.sigmoid(lg_v[best_idx]).item()

                    if best_iou > best_iou_overall:
                        best_iou_overall = best_iou
                        best_level = level

                    # print(f"  Level {level}: best_iou={best_iou:.4f} "
                    #     f"cls_prob={best_score:.4f} "
                    #     f"seg=({left_sec[best_idx]:.1f}, {right_sec[best_idx]:.1f})")

                    # which positions are inside GT?
                    pos_centers = pts[:, 0]  # temporal centers
                    pos_centers_sec = (pos_centers * self.vid_stride * raw['clip_stride']
                                    + 0.5 * raw['clip_size']) / raw['fps']
                    inside_gt = (pos_centers_sec >= gt_start) & (pos_centers_sec <= gt_end)

                    probs_valid = torch.sigmoid(lg_v)

                    if inside_gt.sum() > 0 and (~inside_gt).sum() > 0:
                        pos_prob = probs_valid[inside_gt].mean().item()
                        neg_prob = probs_valid[~inside_gt].mean().item()
                        ratio = pos_prob / (neg_prob + 1e-8)
                        # print(f"  Level {level}: pos_cls={pos_prob:.4f} "
                        #     f"neg_cls={neg_prob:.4f} ratio={ratio:.2f} "
                        #     f"(best_iou={best_iou:.4f})")
                    
                    per_level_info[level] = {
                        'best_iou': best_iou,
                        'cls_prob': best_score,
                        'pos_cls': pos_prob if inside_gt.sum() > 0 else None,
                        'neg_cls': neg_prob if (~inside_gt).sum() > 0 else None,
                    }

                # ---- cross-level context validation ----
                fused = raw['fused_fpn']  # tuple of (B, 256, T_l) per level
                # which fine-level positions are inside GT?
                fine_points = raw['fpn_points'][1]                      # (T_1, 2)
                fine_centers_sec = (fine_points[:, 0] * self.vid_stride * raw['clip_stride']
                                    + 0.5 * raw['clip_size']) / raw['fps']
                inside_gt = (fine_centers_sec >= gt_start) & (fine_centers_sec <= gt_end)
                fine_mask = raw['fpn_masks'][1][0].bool()               # (T_1,)

                valid_inside = inside_gt & fine_mask
                valid_outside = (~inside_gt) & fine_mask

                # replace the single source level check with a multi-source check
                # put this where the cross-level validation code currently is

                best_source_counts = {l: 0 for l in range(2, 6)}  # levels 2-5 as sources
                per_sample_best_gap = []

                for source_level in [2, 3, 4, 5]:
                    coarse_cls = torch.sigmoid(raw['fpn_logits'][source_level][0])
                    coarse_feat = fused[source_level][0]
                    weights = coarse_cls.unsqueeze(0)
                    md = (coarse_feat * weights).sum(dim=1)
                    md = md / (weights.sum() + 1e-8)
                    md = F.normalize(md, dim=0)

                    fine_feat_normed = F.normalize(fused[1][0], dim=0)
                    sim = torch.matmul(md, fine_feat_normed)

                    if valid_inside.sum() > 0 and valid_outside.sum() > 0:
                        gap = sim[valid_inside].mean().item() - sim[valid_outside].mean().item()
                        per_level_info[f'gap_from_L{source_level}'] = gap

                # find best source for this sample
                gaps = {l: per_level_info.get(f'gap_from_L{l}', -999) for l in [2, 3, 4, 5]}
                best_source = max(gaps, key=gaps.get)
                per_level_info['best_source_level'] = best_source
                per_level_info['best_source_gap'] = gaps[best_source]
                # print(f"  Best source: L{best_source} (gap={gaps[best_source]:.4f}) | "
                #     f"all gaps: " + " ".join(f"L{l}={g:.4f}" for l, g in gaps.items()))

                oracle_stats.append({
                    'gt_duration': gt_end - gt_start,
                    'best_level': best_level,
                    'best_iou': best_iou_overall,
                    'per_level': per_level_info,  # add this
                })

            # ---- existing eval logic continues ----

            assert len(results) == len(targets)

            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        
        # ---- oracle summary ----
        import numpy as np
        print("\n" + "=" * 80)
        print("ORACLE LEVEL ANALYSIS SUMMARY")
        for l in range(6):
            entries = [s for s in oracle_stats if s['best_level'] == l]
            if entries:
                avg_iou = np.mean([s['best_iou'] for s in entries])
                avg_dur = np.mean([s['gt_duration'] for s in entries])
                print(f"  Level {l}: {len(entries)} samples ({len(entries)/len(oracle_stats)*100:.1f}%) "
                    f"avg_best_iou={avg_iou:.4f} avg_gt_duration={avg_dur:.1f}s")
        print("\nPER-LEVEL: avg cls_prob of best-IoU proposal")
        for l in range(6):
            ious, probs = [], []
            for s in oracle_stats:
                if l in s['per_level']:
                    ious.append(s['per_level'][l]['best_iou'])
                    probs.append(s['per_level'][l]['cls_prob'])
            if probs:
                ious, probs = np.array(ious), np.array(probs)
                # how often does this level have IoU>0.7 but cls_prob < 0.1?
                good_iou = ious > 0.7
                low_conf = probs < 0.1
                wasted = (good_iou & low_conf).sum()
                print(f"  Level {l}: avg_cls_prob={probs.mean():.4f} "
                    f"avg_iou={ious.mean():.4f} | "
                    f"IoU>0.7 but cls<0.1: {wasted}/{len(probs)} "
                    f"({wasted/len(probs)*100:.1f}%)")
        # In the summary:
        print("\nPER-LEVEL: cls discrimination (pos vs neg)")
        for l in range(6):
            pos_scores, neg_scores = [], []
            for s in oracle_stats:
                if l in s['per_level']:
                    info = s['per_level'][l]
                    if info.get('pos_cls') is not None:
                        pos_scores.append(info['pos_cls'])
                    if info.get('neg_cls') is not None:
                        neg_scores.append(info['neg_cls'])
            if pos_scores:
                pos_m, neg_m = np.mean(pos_scores), np.mean(neg_scores)
                print(f"  Level {l}: avg_pos_cls={pos_m:.4f} "
                    f"avg_neg_cls={neg_m:.4f} "
                    f"ratio={pos_m / (neg_m + 1e-8):.2f}")
        # cross-level context validation summary
        print("\nPER-SAMPLE BEST SOURCE LEVEL:")
        source_counts = {l: 0 for l in [2, 3, 4, 5]}
        source_gaps = {l: [] for l in [2, 3, 4, 5]}
        all_best_gaps = []

        for s in oracle_stats:
            pl = s['per_level']
            if 'best_source_level' in pl:
                bl = pl['best_source_level']
                source_counts[bl] += 1
                all_best_gaps.append(pl['best_source_gap'])
                for l in [2, 3, 4, 5]:
                    if f'gap_from_L{l}' in pl:
                        source_gaps[l].append(pl[f'gap_from_L{l}'])

        total = sum(source_counts.values())
        for l in [2, 3, 4, 5]:
            count = source_counts[l]
            avg_gap = np.mean(source_gaps[l]) if source_gaps[l] else 0
            print(f"  L{l} best for {count}/{total} samples ({count/max(total,1)*100:.1f}%) "
                f"| avg gap when source: {avg_gap:.4f}")

        # compare: using best source per sample vs fixed source
        fixed_L3_gaps = [pl.get('gap_from_L3', 0) for s in oracle_stats 
                        for pl in [s['per_level']] if 'gap_from_L3' in pl]
        print(f"\n  Fixed L3 as source:     avg gap = {np.mean(fixed_L3_gaps):.4f}")
        print(f"  Best source per sample: avg gap = {np.mean(all_best_gaps):.4f}")
        print(f"  Improvement from adaptive source: {np.mean(all_best_gaps) - np.mean(fixed_L3_gaps):.4f}")

        print("\nBEST SOURCE vs GT DURATION:")
        for l in [2, 3, 4, 5]:
            durations = [s['gt_duration'] for s in oracle_stats
                        if s['per_level'].get('best_source_level') == l]
            if durations:
                print(f"  L{l}: avg_duration={np.mean(durations):.1f}s "
                    f"median={np.median(durations):.1f}s "
                    f"range=({np.min(durations):.1f}, {np.max(durations):.1f})")

        # IoU threshold analysis
        for thresh in [0.3, 0.5, 0.7]:
            reachable = sum(1 for s in oracle_stats if s['best_iou'] >= thresh)
            print(f"  Oracle R1@IoU={thresh}: {reachable}/{len(oracle_stats)} "
                f"({reachable/len(oracle_stats)*100:.1f}%)")
            
        print("=" * 80)
        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            # pad video features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _gates, _soft_gates = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            self._all_fpn_raw = []
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):

                # get fused features — handle DDP wrapper
                if hasattr(self.model, 'module'):
                    fused_fpn = self.model.module._last_fused_fpn
                else:
                    fused_fpn = self.model._last_fused_fpn

                self._all_fpn_raw.append({
                    'fpn_points': fpn_points,
                    'fpn_logits': fpn_logits,
                    'fpn_offsets': fpn_offsets,
                    'fpn_masks': fpn_masks,
                    'fused_fpn': fused_fpn,  
                    'clip_stride': data['clip_stride'],
                    'clip_size': data['clip_size'],
                    'fps': data['fps'],
                })
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # ---- print top-k AFTER NMS ----
            # k_post = min(5, len(scores))
            # print(f"Top-{k_post} proposals AFTER NMS:")
            # for i in range(k_post):
            #     print(f"  rank={i:2d} | cls_prob={scores[i]:.4f} | "
            #         f"seg=({segs[i, 0]:.1f}, {segs[i, 1]:.1f})")
            # print("=" * 80)

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _collect_segments_debug(
        self,
        fpn_points, fpn_logits, fpn_offsets, fpn_masks, ext_scores,
    ):
        """Same as _collect_segments but tracks pyramid level per proposal."""
        points_list, scores_list, offsets_list, levels_list = [], [], [], []
        raw_logits_list = []  # pre-sigmoid logits

        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            idx = scores > self.pre_nms_thresh
            points_list.append(points[idx])
            scores_list.append(scores[idx])
            offsets_list.append(offsets[idx])
            raw_logits_list.append(logits[idx])
            levels_list.append(torch.full((idx.sum(),), level, dtype=torch.long, device=points.device))

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        raw_logits = torch.cat(raw_logits_list)
        levels = torch.cat(levels_list)

        # top-k pre-NMS
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]
        raw_logits = raw_logits[idx]
        levels = levels[idx]

        # decode segments (same as original)
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 1]
        right = pt_ctr + offsets[:, 1] * points[:, 1]
        segs = torch.stack([left, right], dim=-1)

        # ---- print top-k BEFORE NMS ----
        k = min(10, len(scores))
        print("=" * 80)
        print(f"Top-{k} proposals BEFORE NMS:")
        for i in range(k):
            print(f"  rank={i:2d} | level={levels[i].item()} | "
                f"cls_prob={scores[i]:.4f} raw_logit={raw_logits[i]:.4f} | "
                f"offset=({offsets[i, 0]:.3f}, {offsets[i, 1]:.3f}) | "
                f"seg=({segs[i, 0]:.1f}, {segs[i, 1]:.1f})")

        # level distribution in top-k
        for l in range(6):
            count = (levels[:k] == l).sum().item()
            if count > 0:
                print(f"  Level {l}: {count}/{k} proposals")
        print("=" * 80)

        return segs, scores
    

    def oracle_level_analysis(fpn_points, fpn_logits, fpn_offsets, fpn_masks, gt_seg):
        """For each level, find the best possible proposal and its IoU with GT."""
        gt_start, gt_end = gt_seg

        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]
            valid = masks.bool()
            if valid.sum() == 0:
                continue

            pts = points[valid]
            off = offsets[valid]

            # decode segments at this level
            pt_ctr = pts[:, 0]
            left = pt_ctr - off[:, 0] * pts[:, 1]
            right = pt_ctr + off[:, 1] * pts[:, 1]

            # IoU with ground truth
            inter_start = torch.clamp(left, min=gt_start)
            inter_end = torch.clamp(right, max=gt_end)
            inter = torch.clamp(inter_end - inter_start, min=0)
            union = (right - left) + (gt_end - gt_start) - inter
            iou = inter / (union + 1e-8)

            best_idx = iou.argmax()
            best_iou = iou[best_idx].item()
            best_score = torch.sigmoid(logits[valid][best_idx]).item()

            print(f"  Level {level}: best_iou={best_iou:.4f} "
                f"cls_prob={best_score:.4f} "
                f"seg=({left[best_idx]:.1f}, {right[best_idx]:.1f})")

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)
        
class EvaluatorWithLog:

    def __init__(self, opt, return_features=False):

        self.opt = opt
        self.return_features = return_features

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        self.split_name = opt['eval']['data']['split']

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformer(opt['model'], return_features=self.return_features).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"{self.split_name}_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        if 'model_ema' in ckpt:
            self.model.load_state_dict(ckpt['model_ema'])
        elif 'best_ema_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['best_ema_state_dict'])
        else:
            raise KeyError(f"No EMA state dict found in {filename}. "
                        f"Available keys: {list(ckpt.keys())}")
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")
        # self.model.load_state_dict(ckpt['model_ema'])
        # print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        import numpy as np
        import csv

        print0("Evaluation started.")
        start_time = time.time()

        # ---- compute FLOPs per forward pass (once) ----
        flops_per_forward = None
        try:
            from fvcore.nn import FlopCountAnalysis
            # build dummy inputs matching padded shapes
            first_data = next(iter(self.dataloader))[0]
            dummy_C = first_data['vid'].size(0)  # feature dim
            dummy_vid = torch.randn(1, dummy_C, self.input_vid_len).cuda()
            dummy_vid_mask = torch.ones(
                1, 1, self.input_vid_len, dtype=torch.bool
            ).cuda()

            # encoder FLOPs
            encoder_flops = FlopCountAnalysis(
                self.model.encode_video, (dummy_vid, dummy_vid_mask)
            )
            encoder_flops.unsupported_ops_warnings(False)
            encoder_flops.uncalled_modules_warnings(False)
            enc_total = encoder_flops.total()

            # fusion + prediction FLOPs (need actual encoder output shapes)
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(dummy_vid, dummy_vid_mask)
                # use a dummy text embedding
                dummy_text = first_data['text']
                if not isinstance(dummy_text, tuple):
                    dummy_text = (dummy_text,)
                dummy_text = dummy_text[0][None].cuda()
                dummy_text_mask = dummy_text.new_full(
                    (1, 1, dummy_text.size(-1)), 1, dtype=torch.bool
                ).cuda()
                dummy_text_enc, dummy_text_mask_enc = \
                    self.model.encode_text(dummy_text, dummy_text_mask)

            fusion_flops = FlopCountAnalysis(
                self.model.fuse_and_predict,
                (fpn, fpn_masks, dummy_text_enc, dummy_text_mask_enc)
            )
            fusion_flops.unsupported_ops_warnings(False)
            fusion_flops.uncalled_modules_warnings(False)
            fuse_total = fusion_flops.total()

            flops_per_forward = enc_total + fuse_total
            print0(f"FLOPs per forward pass: "
                   f"encoder={enc_total/1e9:.2f}G + "
                   f"fusion={fuse_total/1e9:.2f}G = "
                   f"total={flops_per_forward/1e9:.2f}G")
        except ImportError:
            print0("fvcore not installed — skipping FLOP count. "
                   "Install with: pip install fvcore")
        except Exception as e:
            print0(f"FLOP counting failed ({e}), continuing without it.")

        self.flops_per_forward = flops_per_forward

        # ---- per-sample CSV rows ----
        sample_rows = []
        global_sample_idx = 0
        feature_rows = []
        for data_list in self.dataloader:
            results, pred_features = self.predict(data_list[0])
            targets = data_list[0]['segment']

            # extract metadata
            video_duration = data_list[0].get('duration', None)
            video_id = data_list[0].get('video_id',
                       data_list[0].get('vid_id',
                       data_list[0].get('id', None)))
            raw_query = data_list[0].get('query',
                        data_list[0].get('query_text',
                        data_list[0].get('text_raw', None)))
            clip_stride = data_list[0].get('clip_stride', None)
            clip_size = data_list[0].get('clip_size', None)
            fps = data_list[0].get('fps', None)
            vid_len = data_list[0]['vid'].size(-1)

            assert len(results) == len(targets)

            for q_idx, (result, target) in enumerate(zip(results, targets)):
                segs = result['segments']
                scores = result['scores']
                levels = result['levels']
                offsets_raw = result['offsets']

                idx = scores.argsort(descending=True)
                segs = segs[idx[:self.topk]]
                scores = scores[idx[:self.topk]]
                levels = levels[idx[:self.topk]]
                offsets_raw = offsets_raw[idx[:self.topk]]

                target_t = torch.as_tensor(target, dtype=torch.float)
                target_expanded = target_t.expand(len(segs), -1)

                iou_topk = iou(segs, target_expanded)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])

                # ---- build row ----
                gt_start_val = float(target[0])
                gt_end_val = float(target[1])
                gt_dur = gt_end_val - gt_start_val

                r1_iou = iou_topk[:1].max().item() if len(iou_topk) > 0 else 0.0
                r5_iou = iou_topk[:5].max().item() if len(iou_topk) > 0 else 0.0

                n_proposals = len(segs)

                score_gap_r1_r2 = None
                if len(scores) >= 2:
                    score_gap_r1_r2 = (scores[0] - scores[1]).item()

                query_str = None
                if raw_query is not None:
                    if isinstance(raw_query, (list, tuple)):
                        query_str = str(raw_query[q_idx]) if q_idx < len(raw_query) else str(raw_query)
                    elif isinstance(raw_query, str):
                        query_str = raw_query

                row = {
                    'sample_idx': global_sample_idx,
                    'video_id': video_id,
                    'query': query_str,
                    'gt_start': gt_start_val,
                    'gt_end': gt_end_val,
                    'gt_duration': gt_dur,
                    'video_duration': video_duration,
                    'gt_moment_fraction': gt_dur / video_duration if video_duration else None,
                    'vid_feat_len': vid_len,
                    'fps': fps,
                    'clip_stride': clip_stride,
                    'clip_size': clip_size,
                    'r1_iou': r1_iou,
                    'r5_iou': r5_iou,
                    'r5_minus_r1_iou': r5_iou - r1_iou,
                    'n_proposals_after_nms': n_proposals,
                    'score_gap_r1_r2': score_gap_r1_r2,
                    # compute proxies
                    'n_windows': self._n_windows,
                    'query_token_len': self._query_token_lens[q_idx] if q_idx < len(self._query_token_lens) else None,
                    'flops_per_forward': self.flops_per_forward,
                    'total_flops': (self._n_windows * self.flops_per_forward) if self.flops_per_forward else None,
                }

                # top-5 proposal details: level, cls score, iou, segment, offsets
                k = min(5, len(segs))
                for i in range(5):
                    if i < k:
                        row[f'top{i+1}_level'] = levels[i].item()
                        row[f'top{i+1}_score'] = scores[i].item()
                        row[f'top{i+1}_iou'] = iou_topk[i].item()
                        row[f'top{i+1}_seg_start'] = segs[i, 0].item()
                        row[f'top{i+1}_seg_end'] = segs[i, 1].item()
                        row[f'top{i+1}_offset_left'] = offsets_raw[i, 0].item()
                        row[f'top{i+1}_offset_right'] = offsets_raw[i, 1].item()
                    else:
                        row[f'top{i+1}_level'] = None
                        row[f'top{i+1}_score'] = None
                        row[f'top{i+1}_iou'] = None
                        row[f'top{i+1}_seg_start'] = None
                        row[f'top{i+1}_seg_end'] = None
                        row[f'top{i+1}_offset_left'] = None
                        row[f'top{i+1}_offset_right'] = None

                sample_idx_this_query = global_sample_idx
                sample_rows.append(row)

                if self.return_features and pred_features is not None:
                    feat_item = {
                        'sample_idx': sample_idx_this_query,
                        'video_id': video_id,
                        'query': query_str,
                        'split': self.split_name,
                        'features': pred_features[q_idx] if q_idx < len(pred_features) else None,
                    }
                    feature_rows.append(feat_item)

                global_sample_idx += 1

            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()

        # ---- save per-sample CSV ----
        csv_path = os.path.join(
            self.opt['_root'],
            f"per_sample_{self.split_name}_{self.opt['_ckpt']}.csv"
        )
        if sample_rows:
            fieldnames = list(sample_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sample_rows)
            print0(f"\nPer-sample CSV saved to: {csv_path}")
            print0(f"  Total samples: {len(sample_rows)}")
            print0(f"  Columns: {len(fieldnames)}")
        
        if self.return_features and feature_rows:
            feature_path = os.path.join(
                self.opt['_root'],
                f"features_{self.split_name}_{self.opt['_ckpt']}.pt"
            )
            torch.save(feature_rows, feature_path)
            print0(f"Feature file saved to: {feature_path}")
            print0(f"  Total feature rows: {len(feature_rows)}")

        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        query_feature_accumulators = None
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()

        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride

        if n > 0 and n % window_stride > 0:
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        # track per-sample compute proxies
        self._n_windows = len(windows)
        self._query_token_lens = [t.size(-1) for t in text_list]

        segs_list, scores_list = tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()

        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)

            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for q_idx, (text, text_mask) in enumerate(zip(text_list, text_mask_list)):
                fpn_logits, fpn_offsets, _, feat_dict = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )

                if self.return_features:
                    if query_feature_accumulators is None:
                        query_feature_accumulators = [dict() for _ in range(len(text_list))]
                    self._accumulate_query_features(
                        query_feature_accumulators[q_idx],
                        feat_dict,
                        window_offset=window_offset
                    )

            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            window_segs_list, window_scores_list = tuple(), tuple()
            window_levels_list, window_offsets_raw_list = tuple(), tuple()

            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):

                w_segs, w_scores, w_levels, w_offsets_raw = \
                    self._collect_segments(
                        fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                        window_ext[idx] if window_ext is not None else None
                    )
                w_segs += window_offset / self.vid_stride
                window_segs_list += (w_segs.cpu(), )
                window_scores_list += (w_scores.cpu(), )
                window_levels_list += (w_levels.cpu(), )
                window_offsets_raw_list += (w_offsets_raw.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )
            levels_list += (window_levels_list, )
            offsets_raw_list += (window_offsets_raw_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]
        levels_list = [torch.cat(x) for x in zip(*levels_list)]
        offsets_raw_list = [torch.cat(x) for x in zip(*offsets_raw_list)]

        results = tuple()
        for segs, scores, levels, offsets_raw in \
            zip(segs_list, scores_list, levels_list, offsets_raw_list):
            # top-k pre-NMS
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            pre_nms_segs = segs[idx]
            pre_nms_scores = scores[idx]
            pre_nms_levels = levels[idx]
            pre_nms_offsets = offsets_raw[idx]

            # NMS
            post_nms_segs, post_nms_scores = self.batched_nms(
                pre_nms_segs, pre_nms_scores
            )

            # recover level/offset info by matching post-NMS back to pre-NMS
            # (standard NMS preserves segment coordinates exactly)
            if len(post_nms_segs) > 0 and len(pre_nms_segs) > 0:
                diffs = (post_nms_segs.unsqueeze(1)
                         - pre_nms_segs.unsqueeze(0)).abs().sum(-1)
                matched_idx = diffs.argmin(dim=1)
                post_nms_levels = pre_nms_levels[matched_idx]
                post_nms_offsets = pre_nms_offsets[matched_idx]
            else:
                post_nms_levels = torch.zeros(0, dtype=torch.long)
                post_nms_offsets = torch.zeros(0, 2)

            # convert segments to timestamps
            if len(post_nms_segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                post_nms_segs *= self.vid_stride
                post_nms_segs = (post_nms_segs * clip_stride
                                 + 0.5 * clip_size) / fps
                post_nms_segs = torch.clamp(post_nms_segs, min=0, max=duration)

            results += ({
                'segments': post_nms_segs,
                'scores': post_nms_scores,
                'levels': post_nms_levels,
                'offsets': post_nms_offsets,
            }, )
        
        packed_features = None
        if self.return_features:
            packed_features = self._finalize_query_features(query_feature_accumulators)

        return results, packed_features

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()

        # loop over all FPN levels
        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # filter by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )
            offsets_raw_list += (offsets[idx].clone(), )
            levels_list += (
                torch.full((idx.sum(),), level,
                           dtype=torch.long, device=points.device),
            )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        offsets_raw = torch.cat(offsets_raw_list)
        levels = torch.cat(levels_list)

        # only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]
        offsets_raw = offsets_raw[idx]
        levels = levels[idx]

        # assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        # filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]
        levels = levels[idx]
        offsets_raw = offsets_raw[idx]

        return segs, scores, levels, offsets_raw

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)

    def _accumulate_query_features(self, acc, feat_dict, window_offset=0):
        if feat_dict is None:
            return

        for k, v in feat_dict.items():
            if v is None:
                continue
            acc.setdefault(k, []).append(self._to_cpu_detached(v))
    
    def _to_cpu_detached(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        elif isinstance(x, dict):
            return {k: self._to_cpu_detached(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._to_cpu_detached(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(self._to_cpu_detached(v) for v in x)
        else:
            return x
    
    def _finalize_query_features(self, query_feature_accumulators):
        if query_feature_accumulators is None:
            return None

        finalized = []
        for acc in query_feature_accumulators:
            out = {}
            for k, vals in acc.items():
                if len(vals) == 0:
                    out[k] = None
                    continue

                # tensor list면 그냥 list로 두는 게 제일 안전
                # stage마다 shape 다를 수 있어서 억지 stack은 나중에 실패할 수 있음
                out[k] = vals
            finalized.append(out)

        return finalized

class EvaluatorWithLogUsingTrainSet:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['train']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformer(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}_train.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        import numpy as np
        import csv

        print0("Evaluation started.")
        start_time = time.time()

        # ---- compute FLOPs per forward pass (once) ----
        flops_per_forward = None
        try:
            from fvcore.nn import FlopCountAnalysis
            # build dummy inputs matching padded shapes
            first_data = next(iter(self.dataloader))[0]
            dummy_C = first_data['vid'].size(0)  # feature dim
            dummy_vid = torch.randn(1, dummy_C, self.input_vid_len).cuda()
            dummy_vid_mask = torch.ones(
                1, 1, self.input_vid_len, dtype=torch.bool
            ).cuda()

            # encoder FLOPs
            encoder_flops = FlopCountAnalysis(
                self.model.encode_video, (dummy_vid, dummy_vid_mask)
            )
            encoder_flops.unsupported_ops_warnings(False)
            encoder_flops.uncalled_modules_warnings(False)
            enc_total = encoder_flops.total()

            # fusion + prediction FLOPs (need actual encoder output shapes)
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(dummy_vid, dummy_vid_mask)
                # use a dummy text embedding
                dummy_text = first_data['text']
                if not isinstance(dummy_text, tuple):
                    dummy_text = (dummy_text,)
                dummy_text = dummy_text[0][None].cuda()
                dummy_text_mask = dummy_text.new_full(
                    (1, 1, dummy_text.size(-1)), 1, dtype=torch.bool
                ).cuda()
                dummy_text_enc, dummy_text_mask_enc = \
                    self.model.encode_text(dummy_text, dummy_text_mask)

            fusion_flops = FlopCountAnalysis(
                self.model.fuse_and_predict,
                (fpn, fpn_masks, dummy_text_enc, dummy_text_mask_enc)
            )
            fusion_flops.unsupported_ops_warnings(False)
            fusion_flops.uncalled_modules_warnings(False)
            fuse_total = fusion_flops.total()

            flops_per_forward = enc_total + fuse_total
            print0(f"FLOPs per forward pass: "
                   f"encoder={enc_total/1e9:.2f}G + "
                   f"fusion={fuse_total/1e9:.2f}G = "
                   f"total={flops_per_forward/1e9:.2f}G")
        except ImportError:
            print0("fvcore not installed — skipping FLOP count. "
                   "Install with: pip install fvcore")
        except Exception as e:
            print0(f"FLOP counting failed ({e}), continuing without it.")

        self.flops_per_forward = flops_per_forward

        # ---- per-sample CSV rows ----
        sample_rows = []
        global_sample_idx = 0

        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']

            # extract metadata
            video_duration = data_list[0].get('duration', None)
            video_id = data_list[0].get('video_id',
                       data_list[0].get('vid_id',
                       data_list[0].get('id', None)))
            raw_query = data_list[0].get('query',
                        data_list[0].get('query_text',
                        data_list[0].get('text_raw', None)))
            clip_stride = data_list[0].get('clip_stride', None)
            clip_size = data_list[0].get('clip_size', None)
            fps = data_list[0].get('fps', None)
            vid_len = data_list[0]['vid'].size(-1)

            assert len(results) == len(targets)

            for q_idx, (result, target) in enumerate(zip(results, targets)):
                segs = result['segments']
                scores = result['scores']
                levels = result['levels']
                offsets_raw = result['offsets']

                idx = scores.argsort(descending=True)
                segs = segs[idx[:self.topk]]
                scores = scores[idx[:self.topk]]
                levels = levels[idx[:self.topk]]
                offsets_raw = offsets_raw[idx[:self.topk]]

                target_t = torch.as_tensor(target, dtype=torch.float)
                target_expanded = target_t.expand(len(segs), -1)

                iou_topk = iou(segs, target_expanded)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])

                # ---- build row ----
                gt_start_val = float(target[0])
                gt_end_val = float(target[1])
                gt_dur = gt_end_val - gt_start_val

                r1_iou = iou_topk[:1].max().item() if len(iou_topk) > 0 else 0.0
                r5_iou = iou_topk[:5].max().item() if len(iou_topk) > 0 else 0.0

                n_proposals = len(segs)

                score_gap_r1_r2 = None
                if len(scores) >= 2:
                    score_gap_r1_r2 = (scores[0] - scores[1]).item()

                query_str = None
                if raw_query is not None:
                    if isinstance(raw_query, (list, tuple)):
                        query_str = str(raw_query[q_idx]) if q_idx < len(raw_query) else str(raw_query)
                    elif isinstance(raw_query, str):
                        query_str = raw_query

                row = {
                    'sample_idx': global_sample_idx,
                    'video_id': video_id,
                    'query': query_str,
                    'gt_start': gt_start_val,
                    'gt_end': gt_end_val,
                    'gt_duration': gt_dur,
                    'video_duration': video_duration,
                    'gt_moment_fraction': gt_dur / video_duration if video_duration else None,
                    'vid_feat_len': vid_len,
                    'fps': fps,
                    'clip_stride': clip_stride,
                    'clip_size': clip_size,
                    'r1_iou': r1_iou,
                    'r5_iou': r5_iou,
                    'r5_minus_r1_iou': r5_iou - r1_iou,
                    'n_proposals_after_nms': n_proposals,
                    'score_gap_r1_r2': score_gap_r1_r2,
                    # compute proxies
                    'n_windows': self._n_windows,
                    'query_token_len': self._query_token_lens[q_idx] if q_idx < len(self._query_token_lens) else None,
                    'flops_per_forward': self.flops_per_forward,
                    'total_flops': (self._n_windows * self.flops_per_forward) if self.flops_per_forward else None,
                }

                # top-5 proposal details: level, cls score, iou, segment, offsets
                k = min(5, len(segs))
                for i in range(5):
                    if i < k:
                        row[f'top{i+1}_level'] = levels[i].item()
                        row[f'top{i+1}_score'] = scores[i].item()
                        row[f'top{i+1}_iou'] = iou_topk[i].item()
                        row[f'top{i+1}_seg_start'] = segs[i, 0].item()
                        row[f'top{i+1}_seg_end'] = segs[i, 1].item()
                        row[f'top{i+1}_offset_left'] = offsets_raw[i, 0].item()
                        row[f'top{i+1}_offset_right'] = offsets_raw[i, 1].item()
                    else:
                        row[f'top{i+1}_level'] = None
                        row[f'top{i+1}_score'] = None
                        row[f'top{i+1}_iou'] = None
                        row[f'top{i+1}_seg_start'] = None
                        row[f'top{i+1}_seg_end'] = None
                        row[f'top{i+1}_offset_left'] = None
                        row[f'top{i+1}_offset_right'] = None

                sample_rows.append(row)
                global_sample_idx += 1

            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()

        # ---- save per-sample CSV ----
        csv_path = os.path.join(
            self.opt['_root'],
            f"per_sample_eval_{self.opt['_ckpt']}_trainset.csv"
        )
        if sample_rows:
            fieldnames = list(sample_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sample_rows)
            print0(f"\nPer-sample CSV saved to: {csv_path}")
            print0(f"  Total samples: {len(sample_rows)}")
            print0(f"  Columns: {len(fieldnames)}")

        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()

        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride

        if n > 0 and n % window_stride > 0:
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        # track per-sample compute proxies
        self._n_windows = len(windows)
        self._query_token_lens = [t.size(-1) for t in text_list]

        segs_list, scores_list = tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()

        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)

            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            window_segs_list, window_scores_list = tuple(), tuple()
            window_levels_list, window_offsets_raw_list = tuple(), tuple()

            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):

                w_segs, w_scores, w_levels, w_offsets_raw = \
                    self._collect_segments(
                        fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                        window_ext[idx] if window_ext is not None else None
                    )
                w_segs += window_offset / self.vid_stride
                window_segs_list += (w_segs.cpu(), )
                window_scores_list += (w_scores.cpu(), )
                window_levels_list += (w_levels.cpu(), )
                window_offsets_raw_list += (w_offsets_raw.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )
            levels_list += (window_levels_list, )
            offsets_raw_list += (window_offsets_raw_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]
        levels_list = [torch.cat(x) for x in zip(*levels_list)]
        offsets_raw_list = [torch.cat(x) for x in zip(*offsets_raw_list)]

        results = tuple()
        for segs, scores, levels, offsets_raw in \
            zip(segs_list, scores_list, levels_list, offsets_raw_list):
            # top-k pre-NMS
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            pre_nms_segs = segs[idx]
            pre_nms_scores = scores[idx]
            pre_nms_levels = levels[idx]
            pre_nms_offsets = offsets_raw[idx]

            # NMS
            post_nms_segs, post_nms_scores = self.batched_nms(
                pre_nms_segs, pre_nms_scores
            )

            # recover level/offset info by matching post-NMS back to pre-NMS
            # (standard NMS preserves segment coordinates exactly)
            if len(post_nms_segs) > 0 and len(pre_nms_segs) > 0:
                diffs = (post_nms_segs.unsqueeze(1)
                         - pre_nms_segs.unsqueeze(0)).abs().sum(-1)
                matched_idx = diffs.argmin(dim=1)
                post_nms_levels = pre_nms_levels[matched_idx]
                post_nms_offsets = pre_nms_offsets[matched_idx]
            else:
                post_nms_levels = torch.zeros(0, dtype=torch.long)
                post_nms_offsets = torch.zeros(0, 2)

            # convert segments to timestamps
            if len(post_nms_segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                post_nms_segs *= self.vid_stride
                post_nms_segs = (post_nms_segs * clip_stride
                                 + 0.5 * clip_size) / fps
                post_nms_segs = torch.clamp(post_nms_segs, min=0, max=duration)

            results += ({
                'segments': post_nms_segs,
                'scores': post_nms_scores,
                'levels': post_nms_levels,
                'offsets': post_nms_offsets,
            }, )

        return results

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()

        # loop over all FPN levels
        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # filter by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )
            offsets_raw_list += (offsets[idx].clone(), )
            levels_list += (
                torch.full((idx.sum(),), level,
                           dtype=torch.long, device=points.device),
            )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        offsets_raw = torch.cat(offsets_raw_list)
        levels = torch.cat(levels_list)

        # only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]
        offsets_raw = offsets_raw[idx]
        levels = levels[idx]

        # assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        # filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]
        levels = levels[idx]
        offsets_raw = offsets_raw[idx]

        return segs, scores, levels, offsets_raw

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)

class Evaluator:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformer(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        import numpy as np    # move it here

        print0("Evaluation started.")
        start_time = time.time()
        oracle_stats = []
        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']

            # ---- oracle level analysis ----
            raw = self._all_fpn_raw
            per_level_info = {}
            for q_idx, target in enumerate(targets):
                gt_start, gt_end = target[0], target[1]
                raw = self._all_fpn_raw[q_idx]  # per-query data

                print(f"\nQuery {q_idx}: GT=({gt_start:.1f}, {gt_end:.1f}), "
                    f"duration={gt_end - gt_start:.1f}s")

                best_level, best_iou_overall = -1, 0.0
                for level, (points, logits, offsets, masks) in enumerate(
                    zip(raw['fpn_points'], raw['fpn_logits'],
                        raw['fpn_offsets'], raw['fpn_masks'])
                ):
                    lg = logits[0]       # (T_l,)  — already single query
                    off = offsets[0]     # (T_l, 2)
                    mk = masks[0]       # (T_l,)
                    valid = mk.bool()

                    if valid.sum() == 0:
                        print(f"  Level {level}: no valid positions")
                        continue
                    

                    pts = points[valid]       # (N, 2) — center, stride
                    of = off[valid]           # (N, 2)
                    lg_v = lg[valid]          # (N,)

                    # decode to time (same math as _collect_segments)
                    pt_ctr = pts[:, 0]
                    left = pt_ctr - of[:, 0] * pts[:, 1]
                    right = pt_ctr + of[:, 1] * pts[:, 1]

                    # convert feature-space → seconds
                    left_sec = (left * self.vid_stride * raw['clip_stride']
                                + 0.5 * raw['clip_size']) / raw['fps']
                    right_sec = (right * self.vid_stride * raw['clip_stride']
                                + 0.5 * raw['clip_size']) / raw['fps']

                    # IoU with ground truth
                    inter_start = torch.clamp(left_sec, min=gt_start)
                    inter_end = torch.clamp(right_sec, max=gt_end)
                    inter = torch.clamp(inter_end - inter_start, min=0)
                    union = (right_sec - left_sec) + (gt_end - gt_start) - inter
                    iou_vals = inter / (union + 1e-8)

                    best_idx = iou_vals.argmax()
                    best_iou = iou_vals[best_idx].item()
                    best_score = torch.sigmoid(lg_v[best_idx]).item()

                    if best_iou > best_iou_overall:
                        best_iou_overall = best_iou
                        best_level = level

                    print(f"  Level {level}: best_iou={best_iou:.4f} "
                        f"cls_prob={best_score:.4f} "
                        f"seg=({left_sec[best_idx]:.1f}, {right_sec[best_idx]:.1f})")

                    # which positions are inside GT?
                    pos_centers = pts[:, 0]  # temporal centers
                    pos_centers_sec = (pos_centers * self.vid_stride * raw['clip_stride']
                                    + 0.5 * raw['clip_size']) / raw['fps']
                    inside_gt = (pos_centers_sec >= gt_start) & (pos_centers_sec <= gt_end)

                    probs_valid = torch.sigmoid(lg_v)

                    if inside_gt.sum() > 0 and (~inside_gt).sum() > 0:
                        pos_prob = probs_valid[inside_gt].mean().item()
                        neg_prob = probs_valid[~inside_gt].mean().item()
                        ratio = pos_prob / (neg_prob + 1e-8)
                        print(f"  Level {level}: pos_cls={pos_prob:.4f} "
                            f"neg_cls={neg_prob:.4f} ratio={ratio:.2f} "
                            f"(best_iou={best_iou:.4f})")
                    
                    per_level_info[level] = {
                        'best_iou': best_iou,
                        'cls_prob': best_score,
                        'pos_cls': pos_prob if inside_gt.sum() > 0 else None,
                        'neg_cls': neg_prob if (~inside_gt).sum() > 0 else None,
                    }

                # ---- cross-level context validation ----
                fused = raw['fused_fpn']  # tuple of (B, 256, T_l) per level
                # which fine-level positions are inside GT?
                fine_points = raw['fpn_points'][1]                      # (T_1, 2)
                fine_centers_sec = (fine_points[:, 0] * self.vid_stride * raw['clip_stride']
                                    + 0.5 * raw['clip_size']) / raw['fps']
                inside_gt = (fine_centers_sec >= gt_start) & (fine_centers_sec <= gt_end)
                fine_mask = raw['fpn_masks'][1][0].bool()               # (T_1,)

                valid_inside = inside_gt & fine_mask
                valid_outside = (~inside_gt) & fine_mask

                # replace the single source level check with a multi-source check
                # put this where the cross-level validation code currently is

                best_source_counts = {l: 0 for l in range(2, 6)}  # levels 2-5 as sources
                per_sample_best_gap = []

                for source_level in [2, 3, 4, 5]:
                    coarse_cls = torch.sigmoid(raw['fpn_logits'][source_level][0])
                    coarse_feat = fused[source_level][0]
                    weights = coarse_cls.unsqueeze(0)
                    md = (coarse_feat * weights).sum(dim=1)
                    md = md / (weights.sum() + 1e-8)
                    md = F.normalize(md, dim=0)

                    fine_feat_normed = F.normalize(fused[1][0], dim=0)
                    sim = torch.matmul(md, fine_feat_normed)

                    if valid_inside.sum() > 0 and valid_outside.sum() > 0:
                        gap = sim[valid_inside].mean().item() - sim[valid_outside].mean().item()
                        per_level_info[f'gap_from_L{source_level}'] = gap

                # find best source for this sample
                gaps = {l: per_level_info.get(f'gap_from_L{l}', -999) for l in [2, 3, 4, 5]}
                best_source = max(gaps, key=gaps.get)
                per_level_info['best_source_level'] = best_source
                per_level_info['best_source_gap'] = gaps[best_source]
                print(f"  Best source: L{best_source} (gap={gaps[best_source]:.4f}) | "
                    f"all gaps: " + " ".join(f"L{l}={g:.4f}" for l, g in gaps.items()))

                oracle_stats.append({
                    'gt_duration': gt_end - gt_start,
                    'best_level': best_level,
                    'best_iou': best_iou_overall,
                    'per_level': per_level_info,  # add this
                })

            # ---- existing eval logic continues ----

            assert len(results) == len(targets)

            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        
        # ---- oracle summary ----
        import numpy as np
        print("\n" + "=" * 80)
        print("ORACLE LEVEL ANALYSIS SUMMARY")
        for l in range(6):
            entries = [s for s in oracle_stats if s['best_level'] == l]
            if entries:
                avg_iou = np.mean([s['best_iou'] for s in entries])
                avg_dur = np.mean([s['gt_duration'] for s in entries])
                print(f"  Level {l}: {len(entries)} samples ({len(entries)/len(oracle_stats)*100:.1f}%) "
                    f"avg_best_iou={avg_iou:.4f} avg_gt_duration={avg_dur:.1f}s")
        print("\nPER-LEVEL: avg cls_prob of best-IoU proposal")
        for l in range(6):
            ious, probs = [], []
            for s in oracle_stats:
                if l in s['per_level']:
                    ious.append(s['per_level'][l]['best_iou'])
                    probs.append(s['per_level'][l]['cls_prob'])
            if probs:
                ious, probs = np.array(ious), np.array(probs)
                # how often does this level have IoU>0.7 but cls_prob < 0.1?
                good_iou = ious > 0.7
                low_conf = probs < 0.1
                wasted = (good_iou & low_conf).sum()
                print(f"  Level {l}: avg_cls_prob={probs.mean():.4f} "
                    f"avg_iou={ious.mean():.4f} | "
                    f"IoU>0.7 but cls<0.1: {wasted}/{len(probs)} "
                    f"({wasted/len(probs)*100:.1f}%)")
        # In the summary:
        print("\nPER-LEVEL: cls discrimination (pos vs neg)")
        for l in range(6):
            pos_scores, neg_scores = [], []
            for s in oracle_stats:
                if l in s['per_level']:
                    info = s['per_level'][l]
                    if info.get('pos_cls') is not None:
                        pos_scores.append(info['pos_cls'])
                    if info.get('neg_cls') is not None:
                        neg_scores.append(info['neg_cls'])
            if pos_scores:
                pos_m, neg_m = np.mean(pos_scores), np.mean(neg_scores)
                print(f"  Level {l}: avg_pos_cls={pos_m:.4f} "
                    f"avg_neg_cls={neg_m:.4f} "
                    f"ratio={pos_m / (neg_m + 1e-8):.2f}")
        # cross-level context validation summary
        print("\nPER-SAMPLE BEST SOURCE LEVEL:")
        source_counts = {l: 0 for l in [2, 3, 4, 5]}
        source_gaps = {l: [] for l in [2, 3, 4, 5]}
        all_best_gaps = []

        for s in oracle_stats:
            pl = s['per_level']
            if 'best_source_level' in pl:
                bl = pl['best_source_level']
                source_counts[bl] += 1
                all_best_gaps.append(pl['best_source_gap'])
                for l in [2, 3, 4, 5]:
                    if f'gap_from_L{l}' in pl:
                        source_gaps[l].append(pl[f'gap_from_L{l}'])

        total = sum(source_counts.values())
        for l in [2, 3, 4, 5]:
            count = source_counts[l]
            avg_gap = np.mean(source_gaps[l]) if source_gaps[l] else 0
            print(f"  L{l} best for {count}/{total} samples ({count/max(total,1)*100:.1f}%) "
                f"| avg gap when source: {avg_gap:.4f}")

        # compare: using best source per sample vs fixed source
        fixed_L3_gaps = [pl.get('gap_from_L3', 0) for s in oracle_stats 
                        for pl in [s['per_level']] if 'gap_from_L3' in pl]
        print(f"\n  Fixed L3 as source:     avg gap = {np.mean(fixed_L3_gaps):.4f}")
        print(f"  Best source per sample: avg gap = {np.mean(all_best_gaps):.4f}")
        print(f"  Improvement from adaptive source: {np.mean(all_best_gaps) - np.mean(fixed_L3_gaps):.4f}")

        print("\nBEST SOURCE vs GT DURATION:")
        for l in [2, 3, 4, 5]:
            durations = [s['gt_duration'] for s in oracle_stats
                        if s['per_level'].get('best_source_level') == l]
            if durations:
                print(f"  L{l}: avg_duration={np.mean(durations):.1f}s "
                    f"median={np.median(durations):.1f}s "
                    f"range=({np.min(durations):.1f}, {np.max(durations):.1f})")

        # IoU threshold analysis
        for thresh in [0.3, 0.5, 0.7]:
            reachable = sum(1 for s in oracle_stats if s['best_iou'] >= thresh)
            print(f"  Oracle R1@IoU={thresh}: {reachable}/{len(oracle_stats)} "
                f"({reachable/len(oracle_stats)*100:.1f}%)")
            
        print("=" * 80)
        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            # pad video features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            self._all_fpn_raw = []
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):

                # get fused features — handle DDP wrapper
                if hasattr(self.model, 'module'):
                    fused_fpn = self.model.module._last_fused_fpn
                else:
                    fused_fpn = self.model._last_fused_fpn

                self._all_fpn_raw.append({
                    'fpn_points': fpn_points,
                    'fpn_logits': fpn_logits,
                    'fpn_offsets': fpn_offsets,
                    'fpn_masks': fpn_masks,
                    'fused_fpn': fused_fpn,  
                    'clip_stride': data['clip_stride'],
                    'clip_size': data['clip_size'],
                    'fps': data['fps'],
                })
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # ---- print top-k AFTER NMS ----
            # k_post = min(5, len(scores))
            # print(f"Top-{k_post} proposals AFTER NMS:")
            # for i in range(k_post):
            #     print(f"  rank={i:2d} | cls_prob={scores[i]:.4f} | "
            #         f"seg=({segs[i, 0]:.1f}, {segs[i, 1]:.1f})")
            # print("=" * 80)

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _collect_segments_debug(
        self,
        fpn_points, fpn_logits, fpn_offsets, fpn_masks, ext_scores,
    ):
        """Same as _collect_segments but tracks pyramid level per proposal."""
        points_list, scores_list, offsets_list, levels_list = [], [], [], []
        raw_logits_list = []  # pre-sigmoid logits

        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            idx = scores > self.pre_nms_thresh
            points_list.append(points[idx])
            scores_list.append(scores[idx])
            offsets_list.append(offsets[idx])
            raw_logits_list.append(logits[idx])
            levels_list.append(torch.full((idx.sum(),), level, dtype=torch.long, device=points.device))

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        raw_logits = torch.cat(raw_logits_list)
        levels = torch.cat(levels_list)

        # top-k pre-NMS
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]
        raw_logits = raw_logits[idx]
        levels = levels[idx]

        # decode segments (same as original)
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 1]
        right = pt_ctr + offsets[:, 1] * points[:, 1]
        segs = torch.stack([left, right], dim=-1)

        # ---- print top-k BEFORE NMS ----
        k = min(10, len(scores))
        print("=" * 80)
        print(f"Top-{k} proposals BEFORE NMS:")
        for i in range(k):
            print(f"  rank={i:2d} | level={levels[i].item()} | "
                f"cls_prob={scores[i]:.4f} raw_logit={raw_logits[i]:.4f} | "
                f"offset=({offsets[i, 0]:.3f}, {offsets[i, 1]:.3f}) | "
                f"seg=({segs[i, 0]:.1f}, {segs[i, 1]:.1f})")

        # level distribution in top-k
        for l in range(6):
            count = (levels[:k] == l).sum().item()
            if count > 0:
                print(f"  Level {l}: {count}/{k} proposals")
        print("=" * 80)

        return segs, scores
    

    def oracle_level_analysis(fpn_points, fpn_logits, fpn_offsets, fpn_masks, gt_seg):
        """For each level, find the best possible proposal and its IoU with GT."""
        gt_start, gt_end = gt_seg

        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]
            valid = masks.bool()
            if valid.sum() == 0:
                continue

            pts = points[valid]
            off = offsets[valid]

            # decode segments at this level
            pt_ctr = pts[:, 0]
            left = pt_ctr - off[:, 0] * pts[:, 1]
            right = pt_ctr + off[:, 1] * pts[:, 1]

            # IoU with ground truth
            inter_start = torch.clamp(left, min=gt_start)
            inter_end = torch.clamp(right, max=gt_end)
            inter = torch.clamp(inter_end - inter_start, min=0)
            union = (right - left) + (gt_end - gt_start) - inter
            iou = inter / (union + 1e-8)

            best_idx = iou.argmax()
            best_iou = iou[best_idx].item()
            best_score = torch.sigmoid(logits[valid][best_idx]).item()

            print(f"  Level {level}: best_iou={best_iou:.4f} "
                f"cls_prob={best_score:.4f} "
                f"seg=({left[best_idx]:.1f}, {right[best_idx]:.1f})")

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)


class EvaluatorNoNMS: 
    def __init__(self, opt):
 
        self.opt = opt
 
        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))
 
        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0
 
        # load model
        self.model = PtTransformer(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
 
        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))
 
        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride
 
        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size
 
        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
 
        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')
 
        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
 
    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")
 
    @torch.no_grad()
    def run(self):
        import numpy as np
        import csv
        import json
 
        print0("Evaluation started.")
        start_time = time.time()
 
        # ---- compute FLOPs per forward pass (once) ----
        flops_per_forward = None
        try:
            from fvcore.nn import FlopCountAnalysis
            first_data = next(iter(self.dataloader))[0]
            dummy_C = first_data['vid'].size(0)
            dummy_vid = torch.randn(1, dummy_C, self.input_vid_len).cuda()
            dummy_vid_mask = torch.ones(
                1, 1, self.input_vid_len, dtype=torch.bool
            ).cuda()
 
            encoder_flops = FlopCountAnalysis(
                self.model.encode_video, (dummy_vid, dummy_vid_mask)
            )
            encoder_flops.unsupported_ops_warnings(False)
            encoder_flops.uncalled_modules_warnings(False)
            enc_total = encoder_flops.total()
 
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(dummy_vid, dummy_vid_mask)
                dummy_text = first_data['text']
                if not isinstance(dummy_text, tuple):
                    dummy_text = (dummy_text,)
                dummy_text = dummy_text[0][None].cuda()
                dummy_text_mask = dummy_text.new_full(
                    (1, 1, dummy_text.size(-1)), 1, dtype=torch.bool
                ).cuda()
                dummy_text_enc, dummy_text_mask_enc = \
                    self.model.encode_text(dummy_text, dummy_text_mask)
 
            fusion_flops = FlopCountAnalysis(
                self.model.fuse_and_predict,
                (fpn, fpn_masks, dummy_text_enc, dummy_text_mask_enc)
            )
            fusion_flops.unsupported_ops_warnings(False)
            fusion_flops.uncalled_modules_warnings(False)
            fuse_total = fusion_flops.total()
 
            flops_per_forward = enc_total + fuse_total
            print0(f"FLOPs per forward pass: "
                   f"encoder={enc_total/1e9:.2f}G + "
                   f"fusion={fuse_total/1e9:.2f}G = "
                   f"total={flops_per_forward/1e9:.2f}G")
        except ImportError:
            print0("fvcore not installed — skipping FLOP count. "
                   "Install with: pip install fvcore")
        except Exception as e:
            print0(f"FLOP counting failed ({e}), continuing without it.")
        
        print("No NMS")
 
        self.flops_per_forward = flops_per_forward
 
        # ---- per-sample CSV rows + no-NMS diagnostic ----
        sample_rows = []
        no_nms_results = []
        global_sample_idx = 0
 
        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']
 
            # extract metadata
            video_duration = data_list[0].get('duration', None)
            video_id = data_list[0].get('video_id',
                       data_list[0].get('vid_id',
                       data_list[0].get('id', None)))
            raw_query = data_list[0].get('query',
                        data_list[0].get('query_text',
                        data_list[0].get('text_raw', None)))
            clip_stride = data_list[0].get('clip_stride', None)
            clip_size = data_list[0].get('clip_size', None)
            fps = data_list[0].get('fps', None)
            vid_len = data_list[0]['vid'].size(-1)
 
            assert len(results) == len(targets)
 
            for q_idx, (result, target) in enumerate(zip(results, targets)):
                segs = result['segments']
                scores = result['scores']
                levels = result['levels']
                offsets_raw = result['offsets']
 
                idx = scores.argsort(descending=True)
                segs = segs[idx[:self.topk]]
                scores = scores[idx[:self.topk]]
                levels = levels[idx[:self.topk]]
                offsets_raw = offsets_raw[idx[:self.topk]]
 
                target_t = torch.as_tensor(target, dtype=torch.float)
                target_expanded = target_t.expand(len(segs), -1)
 
                iou_topk = iou(segs, target_expanded)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
 
                # ---- build per-sample CSV row ----
                gt_start_val = float(target[0])
                gt_end_val = float(target[1])
                gt_dur = gt_end_val - gt_start_val
 
                r1_iou = iou_topk[:1].max().item() if len(iou_topk) > 0 else 0.0
                r5_iou = iou_topk[:5].max().item() if len(iou_topk) > 0 else 0.0
 
                n_proposals = len(segs)
 
                score_gap_r1_r2 = None
                if len(scores) >= 2:
                    score_gap_r1_r2 = (scores[0] - scores[1]).item()
 
                query_str = None
                if raw_query is not None:
                    if isinstance(raw_query, (list, tuple)):
                        query_str = str(raw_query[q_idx]) if q_idx < len(raw_query) else str(raw_query)
                    elif isinstance(raw_query, str):
                        query_str = raw_query
 
                row = {
                    'sample_idx': global_sample_idx,
                    'video_id': video_id,
                    'query': query_str,
                    'gt_start': gt_start_val,
                    'gt_end': gt_end_val,
                    'gt_duration': gt_dur,
                    'video_duration': video_duration,
                    'gt_moment_fraction': gt_dur / video_duration if video_duration else None,
                    'vid_feat_len': vid_len,
                    'fps': fps,
                    'clip_stride': clip_stride,
                    'clip_size': clip_size,
                    'r1_iou': r1_iou,
                    'r5_iou': r5_iou,
                    'r5_minus_r1_iou': r5_iou - r1_iou,
                    'n_proposals_after_nms': n_proposals,
                    'score_gap_r1_r2': score_gap_r1_r2,
                    'n_windows': self._n_windows,
                    'query_token_len': self._query_token_lens[q_idx] if q_idx < len(self._query_token_lens) else None,
                    'flops_per_forward': self.flops_per_forward,
                    'total_flops': (self._n_windows * self.flops_per_forward) if self.flops_per_forward else None,
                }
 
                # top-5 proposal details
                k = min(5, len(segs))
                for i in range(5):
                    if i < k:
                        row[f'top{i+1}_level'] = levels[i].item()
                        row[f'top{i+1}_score'] = scores[i].item()
                        row[f'top{i+1}_iou'] = iou_topk[i].item()
                        row[f'top{i+1}_seg_start'] = segs[i, 0].item()
                        row[f'top{i+1}_seg_end'] = segs[i, 1].item()
                        row[f'top{i+1}_offset_left'] = offsets_raw[i, 0].item()
                        row[f'top{i+1}_offset_right'] = offsets_raw[i, 1].item()
                    else:
                        row[f'top{i+1}_level'] = None
                        row[f'top{i+1}_score'] = None
                        row[f'top{i+1}_iou'] = None
                        row[f'top{i+1}_seg_start'] = None
                        row[f'top{i+1}_seg_end'] = None
                        row[f'top{i+1}_offset_left'] = None
                        row[f'top{i+1}_offset_right'] = None
 
                # ---- No-NMS diagnostic: top-1/top-5 by cls_score ----
                pre_nms = self._pre_nms_proposals[q_idx]
                pre_segs_grid = pre_nms['segs_grid']
                pre_scores = pre_nms['scores']
                pre_levels = pre_nms['levels']
 
                if len(pre_scores) == 0:
                    no_nms_row = {
                        'sample_idx': global_sample_idx,
                        'gt_sec': [gt_start_val, gt_end_val],
                        'gt_duration': gt_dur,
                        'duration': float(video_duration) if video_duration else None,
                        'top1_iou': 0.0, 'top1_score': 0.0, 'top1_level': -1,
                        'top1_seg_sec': [0.0, 0.0],
                        'top5_best_iou': 0.0, 'top5_ious': [], 'top5_levels': [],
                        'num_proposals': 0,
                    }
                else:
                    # Convert grid → seconds
                    nn_segs_sec = pre_segs_grid.clone().float() * self.vid_stride
                    nn_segs_sec = (nn_segs_sec * clip_stride + 0.5 * clip_size) / fps
                    duration_val = float(video_duration) if video_duration else 1e8
                    nn_segs_sec = torch.clamp(nn_segs_sec, min=0, max=duration_val)
 
                    # IoU with GT
                    nn_inter_s = torch.clamp(nn_segs_sec[:, 0], min=gt_start_val)
                    nn_inter_e = torch.clamp(nn_segs_sec[:, 1], max=gt_end_val)
                    nn_inter = torch.clamp(nn_inter_e - nn_inter_s, min=0)
                    nn_union = (nn_segs_sec[:, 1] - nn_segs_sec[:, 0]) + gt_dur - nn_inter
                    nn_ious = nn_inter / (nn_union + 1e-8)
 
                    # Already sorted by score descending
                    nn_top1_iou = nn_ious[0].item()
                    nn_top1_score = pre_scores[0].item()
                    nn_top1_level = pre_levels[0].item()
                    nn_top1_seg = nn_segs_sec[0].tolist()
 
                    nn_k = min(5, len(nn_ious))
                    nn_top5_ious = nn_ious[:nn_k].tolist()
                    nn_top5_levels = pre_levels[:nn_k].tolist()
                    nn_top5_best_iou = max(nn_top5_ious)
 
                    no_nms_row = {
                        'sample_idx': global_sample_idx,
                        'gt_sec': [gt_start_val, gt_end_val],
                        'gt_duration': gt_dur,
                        'duration': float(video_duration) if video_duration else None,
                        'top1_iou': nn_top1_iou,
                        'top1_score': nn_top1_score,
                        'top1_level': nn_top1_level,
                        'top1_seg_sec': nn_top1_seg,
                        'top5_best_iou': nn_top5_best_iou,
                        'top5_ious': nn_top5_ious,
                        'top5_levels': nn_top5_levels,
                        'num_proposals': len(pre_scores),
                    }
 
                # Add no-NMS info to CSV row too
                row['no_nms_top1_iou'] = no_nms_row['top1_iou']
                row['no_nms_top1_score'] = no_nms_row['top1_score']
                row['no_nms_top1_level'] = no_nms_row['top1_level']
                row['no_nms_top5_best_iou'] = no_nms_row['top5_best_iou']
                row['no_nms_num_proposals'] = no_nms_row['num_proposals']
 
                sample_rows.append(row)
                no_nms_results.append(no_nms_row)
                global_sample_idx += 1
 
            self.text_cnt += len(targets)
            self.itr += 1
 
            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
 
        # ---- save per-sample CSV ----
        csv_path = os.path.join(
            self.opt['_root'],
            f"per_sample_eval_{self.opt['_ckpt']}.csv"
        )
        if sample_rows:
            fieldnames = list(sample_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sample_rows)
            print0(f"\nPer-sample CSV saved to: {csv_path}")
            print0(f"  Total samples: {len(sample_rows)}")
            print0(f"  Columns: {len(fieldnames)}")
 
        # ---- No-NMS diagnostic summary ----
        print("\n" + "=" * 80)
        print("NO-NMS DIAGNOSTIC (top-1 / top-5 by cls_score, no NMS)")
        print("=" * 80)
 
        nn_top1_ious = np.array([r['top1_iou'] for r in no_nms_results])
        nn_top5_ious = np.array([r['top5_best_iou'] for r in no_nms_results])
 
        # Also collect with-NMS for side-by-side comparison
        nms_r1_ious = np.array([r['r1_iou'] for r in sample_rows])
        nms_r5_ious = np.array([r['r5_iou'] for r in sample_rows])
 
        print(f"\n{'':>20s}  {'No-NMS':>12s}  {'With-NMS':>12s}  {'Diff':>8s}")
        print(f"{'':>20s}  {'─'*12}  {'─'*12}  {'─'*8}")
        for thresh in [0.3, 0.5, 0.7]:
            nn_r1 = (nn_top1_ious >= thresh).mean() * 100
            nn_r5 = (nn_top5_ious >= thresh).mean() * 100
            nms_r1 = (nms_r1_ious >= thresh).mean() * 100
            nms_r5 = (nms_r5_ious >= thresh).mean() * 100
            print(f"  R1@IoU={thresh}:       {nn_r1:6.2f}%       {nms_r1:6.2f}%    {nms_r1 - nn_r1:+.2f}")
            print(f"  R5@IoU={thresh}:       {nn_r5:6.2f}%       {nms_r5:6.2f}%    {nms_r5 - nn_r5:+.2f}")
 
        # Top-1 level distribution (no-NMS)
        print(f"\nNo-NMS top-1 level distribution:")
        all_levels = [r['top1_level'] for r in no_nms_results]
        max_level = max(all_levels) if all_levels else 0
        for l in range(max_level + 1):
            count = all_levels.count(l)
            if count > 0:
                avg_iou = np.mean([r['top1_iou'] for r in no_nms_results
                                   if r['top1_level'] == l])
                print(f"  Level {l}: {count}/{len(no_nms_results)} "
                      f"({count/len(no_nms_results)*100:.1f}%)  "
                      f"avg_iou={avg_iou:.4f}")
 
        # Per-sample: no-NMS vs with-NMS comparison
        improved_by_nms = 0
        degraded_by_nms = 0
        for row in sample_rows:
            diff = row['r1_iou'] - row['no_nms_top1_iou']
            if diff > 0.01:
                improved_by_nms += 1
            elif diff < -0.01:
                degraded_by_nms += 1
        print(f"\nNMS impact on top-1 IoU:")
        print(f"  Improved by NMS:  {improved_by_nms}/{len(sample_rows)} "
              f"({improved_by_nms/len(sample_rows)*100:.1f}%)")
        print(f"  Degraded by NMS:  {degraded_by_nms}/{len(sample_rows)} "
              f"({degraded_by_nms/len(sample_rows)*100:.1f}%)")
        print(f"  Unchanged:        {len(sample_rows) - improved_by_nms - degraded_by_nms}/{len(sample_rows)}")
 
        # Save no-NMS JSON
        no_nms_path = os.path.join(
            self.opt['_root'],
            f"no_nms_diagnostic_{self.opt['_ckpt']}.json"
        )
        with open(no_nms_path, 'w') as f:
            json.dump(no_nms_results, f, indent=2)
        print(f"\nNo-NMS diagnostic saved to: {no_nms_path}")
 
        print("=" * 80)
        self.log(is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")
 
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
 
        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)
 
            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )
 
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
 
        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]
 
        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size
 
        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
 
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
 
        if n > 0 and n % window_stride > 0:
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )
 
        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride
 
        # track per-sample compute proxies
        self._n_windows = len(windows)
        self._query_token_lens = [t.size(-1) for t in text_list]
 
        segs_list, scores_list = tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()
 
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
 
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
 
            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]
 
            window_segs_list, window_scores_list = tuple(), tuple()
            window_levels_list, window_offsets_raw_list = tuple(), tuple()
 
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
 
                w_segs, w_scores, w_levels, w_offsets_raw = \
                    self._collect_segments(
                        fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                        window_ext[idx] if window_ext is not None else None
                    )
                w_segs += window_offset / self.vid_stride
                window_segs_list += (w_segs.cpu(), )
                window_scores_list += (w_scores.cpu(), )
                window_levels_list += (w_levels.cpu(), )
                window_offsets_raw_list += (w_offsets_raw.cpu(), )
 
            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )
            levels_list += (window_levels_list, )
            offsets_raw_list += (window_offsets_raw_list, )
 
        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]
        levels_list = [torch.cat(x) for x in zip(*levels_list)]
        offsets_raw_list = [torch.cat(x) for x in zip(*offsets_raw_list)]
 
        # ---- Save pre-NMS proposals for no-NMS diagnostic ----
        self._pre_nms_proposals = []
 
        results = tuple()
        for segs, scores, levels, offsets_raw in \
            zip(segs_list, scores_list, levels_list, offsets_raw_list):
            # top-k pre-NMS
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            pre_nms_segs = segs[idx]
            pre_nms_scores = scores[idx]
            pre_nms_levels = levels[idx]
            pre_nms_offsets = offsets_raw[idx]
 
            # Store for no-NMS diagnostic (already sorted by score desc)
            self._pre_nms_proposals.append({
                'segs_grid': pre_nms_segs.clone(),
                'scores': pre_nms_scores.clone(),
                'levels': pre_nms_levels.clone(),
            })
 
            # NMS
            post_nms_segs, post_nms_scores = self.batched_nms(
                pre_nms_segs, pre_nms_scores
            )
 
            # recover level/offset info by matching post-NMS back to pre-NMS
            if len(post_nms_segs) > 0 and len(pre_nms_segs) > 0:
                diffs = (post_nms_segs.unsqueeze(1)
                         - pre_nms_segs.unsqueeze(0)).abs().sum(-1)
                matched_idx = diffs.argmin(dim=1)
                post_nms_levels = pre_nms_levels[matched_idx]
                post_nms_offsets = pre_nms_offsets[matched_idx]
            else:
                post_nms_levels = torch.zeros(0, dtype=torch.long)
                post_nms_offsets = torch.zeros(0, 2)
 
            # convert segments to timestamps
            if len(post_nms_segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']
 
                post_nms_segs *= self.vid_stride
                post_nms_segs = (post_nms_segs * clip_stride
                                 + 0.5 * clip_size) / fps
                post_nms_segs = torch.clamp(post_nms_segs, min=0, max=duration)
 
            results += ({
                'segments': post_nms_segs,
                'scores': post_nms_scores,
                'levels': post_nms_levels,
                'offsets': post_nms_offsets,
            }, )
 
        return results
 
    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()
        levels_list, offsets_raw_list = tuple(), tuple()
 
        # loop over all FPN levels
        for level, (points, logits, offsets, masks) in enumerate(
            zip(fpn_points, fpn_logits, fpn_offsets, fpn_masks)
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]
 
            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()
 
            # filter by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )
            offsets_raw_list += (offsets[idx].clone(), )
            levels_list += (
                torch.full((idx.sum(),), level,
                           dtype=torch.long, device=points.device),
            )
 
        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)
        offsets_raw = torch.cat(offsets_raw_list)
        levels = torch.cat(levels_list)
 
        # only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]
        offsets_raw = offsets_raw[idx]
        levels = levels[idx]
 
        # assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)
 
        # filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]
        levels = levels[idx]
        offsets_raw = offsets_raw[idx]
 
        return segs, scores, levels, offsets_raw
 
    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)



class EvaluatorAnalysis:

    def __init__(self, opt):
        # ... existing init code ...
        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformer(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        # Add storage for attention analysis
        self.attn_analysis_results = []
        self.n_decoder_layers = opt['model']['fusion'].get('n_layers', 2)

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu')
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        """Run evaluation with attention entropy analysis."""
        print0("Attention analysis started.")
        
        for data_list in self.dataloader:
            data = data_list[0]
            
            # Get predictions + attention weights
            results, attn_data = self.predict_with_attn(data)
            targets = data['segment']
            
            for query_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                if len(segs) == 0:
                    continue
                
                # Get best prediction's tIoU
                best_idx = scores.argmax()
                best_seg = segs[best_idx].unsqueeze(0)
                target_t = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
                tiou = iou(best_seg, target_t).item()
                
                # Get attention weights for this query
                # attn_data[query_idx] is a dict with per-level, per-layer weights
                query_attn = attn_data[query_idx]
                
                entry = {
                    'tiou': tiou,
                    'query_length': data['text'][query_idx].size(-1) 
                                    if isinstance(data['text'], tuple) 
                                    else data['text'].size(-1),
                }
                
                for key, attn_w in query_attn.items():
                    # attn_w: (1, h, t1, t2)
                    entropy = self._compute_entropy(attn_w)
                    confidence = self._compute_confidence(attn_w)
                    sparsity = self._compute_sparsity(attn_w)
                    
                    entry[f'{key}/entropy'] = entropy
                    entry[f'{key}/confidence'] = confidence
                    entry[f'{key}/sparsity'] = sparsity
                
                self.attn_analysis_results.append(entry)
            
            self.itr += 1
            if self.itr % self.log_interval == 0:
                print0(f"[{self.itr}/{self.num_itrs}] "
                       f"Collected {len(self.attn_analysis_results)} samples")
        
        self._save_and_plot_analysis()

    def predict_with_attn(self, data):
        """Same as predict() but also returns attention weights per query."""
        
        # ====== Text encoding (same as original) ======
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens,)
        
        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)
            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text,)
            text_mask_list += (text_mask,)
        
        # ====== Video encoding (same as original) ======
        vid = data['vid']
        vid_len = vid.size(-1)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]
        
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size
        
        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size],)
            window_offsets += (idx,)
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size],)
            else:
                window_ext_scores += (None,)
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            windows += (vid[..., -window_size:],)
            window_offsets += (n,)
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:],)
            else:
                window_ext_scores += (None,)
        
        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride
        
        # ====== Fusion with attention capture ======
        # We only capture attention from the FIRST window for simplicity
        # (Charades-STA videos are short, typically 1 window)
        
        segs_list, scores_list = tuple(), tuple()
        per_query_attn = {i: {} for i in range(len(text_list))}
        
        for w_idx, (window, window_offset, window_ext) in enumerate(
            zip(windows, window_offsets, window_ext_scores)
        ):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            
            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for q_idx, (text, text_mask) in enumerate(
                zip(text_list, text_mask_list)
            ):
                # *** Clear attention weights before each fusion call ***
                self.model.fusion._last_attn_weights = []
                
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits,)
                fpn_offsets_list += (fpn_offsets,)
                
                # *** Capture attention weights (first window only) ***
                if w_idx == 0:
                    attn_list = self.model.fusion._last_attn_weights
                    num_levels = len(attn_list) // self.n_decoder_layers
                    
                    for level_idx in range(num_levels):
                        for layer_idx in range(self.n_decoder_layers):
                            flat_idx = level_idx * self.n_decoder_layers + layer_idx
                            if flat_idx < len(attn_list):
                                key = f'level{level_idx}_layer{layer_idx}'
                                per_query_attn[q_idx][key] = \
                                    attn_list[flat_idx].cpu()
            
            fpn_masks_squeezed = [m.squeeze(1) for m in fpn_masks]
            
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in enumerate(
                zip(fpn_logits_list, fpn_offsets_list)
            ):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks_squeezed,
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(),)
                window_scores_list += (window_scores.cpu(),)
            
            segs_list += (window_segs_list,)
            scores_list += (window_scores_list,)
        
        # ====== Assemble results (same as original) ======
        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]
        
        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            segs, scores = self.batched_nms(segs[idx], scores[idx])
            
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']
                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)
            
            results += ({'segments': segs, 'scores': scores},)
        
        return results, per_query_attn

    # ====== Metric computation ======
    
    def _compute_entropy(self, attn_w):
        """
        attn_w: (1, h, t1, t2)
        Returns scalar normalized entropy.
        """
        attn = attn_w.squeeze(0).float()  # (h, t1, t2)
        log_attn = (attn + 1e-8).log()
        entropy = -(attn * log_attn).sum(dim=-1)  # (h, t1)
        
        # Normalize by max possible entropy
        num_tokens = attn.size(-1)
        max_entropy = np.log(num_tokens)
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy.mean().item()

    def _compute_confidence(self, attn_w):
        """Max attention weight, averaged across heads and positions."""
        attn = attn_w.squeeze(0).float()  # (h, t1, t2)
        return attn.max(dim=-1).values.mean().item()

    def _compute_sparsity(self, attn_w, threshold=0.1):
        """Fraction of tokens receiving > threshold attention."""
        attn = attn_w.squeeze(0).float()  # (h, t1, t2)
        active = (attn > threshold).float()
        return (active.sum(dim=-1) / attn.size(-1)).mean().item()
    def _compute_attention_stats(self, attn_w):
        """Check if attention is degenerate."""
        attn = attn_w.squeeze(0).float()  # (h, t1, t2)
        
        stats = {
            'mean_entropy': None,
            'mean_max': None,
            'is_near_uniform': None,
            'is_near_onehot': None,
        }
        
        num_tokens = attn.size(-1)
        max_entropy = np.log(num_tokens)
        
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)  # (h, t1)
        normalized_entropy = (entropy / max_entropy).mean().item()
        
        max_val = attn.max(dim=-1).values.mean().item()
        
        stats['mean_entropy'] = normalized_entropy
        stats['mean_max'] = max_val
        stats['is_near_uniform'] = normalized_entropy > 0.85
        stats['is_near_onehot'] = max_val > 0.7
        
        return stats

    # ====== Visualization ======

    def _save_and_plot_analysis(self):
        import json
        import matplotlib.pyplot as plt
        
        save_dir = os.path.join(self.opt['_root'], 'attention_analysis')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw data
        with open(os.path.join(save_dir, 'raw_results.json'), 'w') as f:
            json.dump(self.attn_analysis_results, f, indent=2)
        
        # Identify all attention keys from first entry
        sample = self.attn_analysis_results[0]
        attn_keys = sorted(set(
            k.rsplit('/', 1)[0] for k in sample.keys() 
            if '/' in k
        ))
        
        tious = np.array([r['tiou'] for r in self.attn_analysis_results])
        
        # ====== Plot 1: Scatter plots per level/layer ======
        n_keys = len(attn_keys)
        n_cols = min(4, n_keys)
        n_rows = (n_keys + n_cols - 1) // n_cols
        
        for metric in ['entropy', 'confidence', 'sparsity']:
            fig, axes = plt.subplots(n_rows, n_cols, 
                                     figsize=(5 * n_cols, 4 * n_rows))
            if n_keys == 1:
                axes = np.array([[axes]])
            axes = axes.flatten()
            
            for idx, key in enumerate(attn_keys):
                full_key = f'{key}/{metric}'
                values = np.array([r[full_key] for r in self.attn_analysis_results])
                
                ax = axes[idx]
                ax.scatter(values, tious, alpha=0.2, s=8)
                ax.set_xlabel(f'{metric}')
                ax.set_ylabel('tIoU')
                ax.set_title(key)
                
                # Correlation
                corr = np.corrcoef(values, tious)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        fontsize=11, fontweight='bold',
                        color='green' if abs(corr) > 0.2 else 'red')
            
            # Hide unused axes
            for idx in range(n_keys, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'{metric.capitalize()} vs tIoU', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric}_vs_tiou.png'), dpi=150)
            plt.close()
        
        # ====== Plot 2: Binned analysis (most important) ======
        fig, axes = plt.subplots(1, len(attn_keys), 
                                 figsize=(4 * len(attn_keys), 5))
        if len(attn_keys) == 1:
            axes = [axes]
        
        for idx, key in enumerate(attn_keys):
            entropies = np.array([
                r[f'{key}/entropy'] for r in self.attn_analysis_results
            ])
            
            # Bin into quartiles
            bins = np.percentile(entropies, [0, 25, 50, 75, 100])
            labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            means, stds = [], []
            
            for b in range(4):
                mask = (entropies >= bins[b]) & (entropies <= bins[b + 1])
                if b < 3:
                    mask = (entropies >= bins[b]) & (entropies < bins[b + 1])
                means.append(tious[mask].mean())
                stds.append(tious[mask].std() / np.sqrt(mask.sum()))
            
            axes[idx].bar(labels, means, yerr=stds, color=colors, 
                         capsize=4, edgecolor='black', linewidth=0.5)
            axes[idx].set_ylabel('Mean tIoU')
            axes[idx].set_xlabel('Entropy Quartile')
            axes[idx].set_title(key)
        
        plt.suptitle('Mean tIoU by Attention Entropy Quartile', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'entropy_quartiles.png'), dpi=150)
        plt.close()
        
        # ====== Print summary ======
        print0("\n" + "=" * 60)
        print0("ATTENTION ANALYSIS SUMMARY")
        print0("=" * 60)
        print0(f"Total samples: {len(self.attn_analysis_results)}")
        print0(f"Mean tIoU: {tious.mean():.4f}")
        print0("")
        
        for key in attn_keys:
            entropies = np.array([
                r[f'{key}/entropy'] for r in self.attn_analysis_results
            ])
            confidences = np.array([
                r[f'{key}/confidence'] for r in self.attn_analysis_results
            ])
            
            corr_ent = np.corrcoef(entropies, tious)[0, 1]
            corr_conf = np.corrcoef(confidences, tious)[0, 1]
            
            print0(f"{key}:")
            print0(f"  entropy  → tIoU correlation: {corr_ent:+.4f}"
                   f"  {'*** STRONG' if abs(corr_ent) > 0.3 else '* weak' if abs(corr_ent) > 0.15 else '  none'}")
            print0(f"  confidence → tIoU correlation: {corr_conf:+.4f}"
                   f"  {'*** STRONG' if abs(corr_conf) > 0.3 else '* weak' if abs(corr_conf) > 0.15 else '  none'}")
        
        print0("=" * 60)

        # After collecting all results, add this summary
        for key in attn_keys:
            all_entropies = [r[f'{key}/entropy'] for r in self.attn_analysis_results]
            all_confidences = [r[f'{key}/confidence'] for r in self.attn_analysis_results]
            
            mean_ent = np.mean(all_entropies)
            std_ent = np.std(all_entropies)
            mean_conf = np.mean(all_confidences)
            
            print0(f"{key}:")
            print0(f"  entropy:    mean={mean_ent:.4f}, std={std_ent:.4f}")
            print0(f"  confidence: mean={mean_conf:.4f}")
            
            if mean_ent > 0.85:
                print0(f"  → NEAR UNIFORM — attention is spread across all tokens")
            elif mean_conf > 0.6:
                print0(f"  → PEAKED — attention focuses on 1-2 tokens")
            else:
                print0(f"  → MODERATE — attention is selective but not extreme")
    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores
    
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        for text in tokens:
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)

            text, text_mask = self.model.encode_text(text, text_mask)
            text_list += (text, )
            text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            # pad video features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results
    
    @torch.no_grad()
    def run_token_attention_analysis(self):
        """Check WHICH text tokens the model attends to,
        and whether attending to the right tokens predicts success."""
        
        records = []
        
        for data_list in self.dataloader:
            data = data_list[0]
            
            tokens = data['text']
            if not isinstance(tokens, tuple):
                tokens = (tokens,)
            
            # Get raw token strings if available
            # If not, we'll work with token indices
            raw_queries = data.get('raw_text', None)
            
            # Encode text
            text_list, text_mask_list = [], []
            for text in tokens:
                text_t = text[None].cuda(non_blocking=True)
                text_mask = text_t.new_full(
                    (1, 1, text_t.size(-1)), 1, dtype=torch.bool
                )
                text_enc, text_mask_enc = self.model.encode_text(text_t, text_mask)
                text_list.append(text_enc)
                text_mask_list.append(text_mask_enc)
            
            # Encode video
            vid = data['vid']
            vid_len = vid.size(-1)
            input_vid_len = self.input_vid_len
            window = F.pad(vid, (0, max(0, input_vid_len - vid_len)))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < vid_len
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            
            fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            
            targets = data['segment']
            
            for q_idx, (text, text_mask) in enumerate(
                zip(text_list, text_mask_list)
            ):
                # Clear and run fusion
                self.model.fusion._last_attn_weights = []
                fpn_logits, fpn_offsets, _, _ = \
                    self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                
                attn_list = self.model.fusion._last_attn_weights
                
                # Get prediction
                fpn_masks_sq = [m.squeeze(1) for m in fpn_masks]
                segs, scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks_sq, None
                )
                
                if len(segs) == 0:
                    continue
                
                # *** Convert segments from feature-space to timestamps ***
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']
                
                segs_time = segs.clone()
                segs_time *= self.vid_stride
                segs_time = (segs_time * clip_stride + 0.5 * clip_size) / fps
                segs_time = torch.clamp(segs_time, min=0, max=duration)
                
                target = targets[q_idx] if len(targets) > 1 else targets[0]
                target_t = torch.as_tensor(target, dtype=torch.float).to(segs_time.device)
                best_idx = scores.argmax()
                tiou = iou(segs_time[best_idx].unsqueeze(0),
                        target_t.unsqueeze(0)).item()
                
                gt_start, gt_end = target[0], target[1]
                
                # For indexing into attention maps, convert GT timestamps
                # to feature-space positions
                vid_len_feat = fpn_masks[0].sum().item()  # valid positions at level 0
                
                # Analyze attention at level 0 (finest), both layers
                # We want: what tokens does the GT region attend to
                # vs what tokens does the predicted region attend to
                
                num_levels = len(attn_list) // self.n_decoder_layers
                
                for layer_idx in range(self.n_decoder_layers):
                    attn_w = attn_list[
                        0 * self.n_decoder_layers + layer_idx
                    ]  # (1, h, T_vid, T_text)
                    
                    attn_avg = attn_w.squeeze(0).mean(dim=0)  # (T_vid, T_text)
                    T_vid, T_text = attn_avg.shape
                    
                    # Convert GT timestamps to feature-space indices
                    # GT is in seconds, feature positions correspond to clip centers
                    gt_start_feat = (gt_start * fps - 0.5 * clip_size) / clip_stride
                    gt_end_feat = (gt_end * fps - 0.5 * clip_size) / clip_stride
                    gt_start_feat /= self.vid_stride
                    gt_end_feat /= self.vid_stride
                    
                    gt_start_idx = max(0, int(gt_start_feat))
                    gt_end_idx = min(T_vid, int(gt_end_feat) + 1)
                    
                    if gt_start_idx >= gt_end_idx:
                        gt_start_idx = max(0, gt_end_idx - 1)
                    
                    gt_attn = attn_avg[gt_start_idx:gt_end_idx].mean(dim=0)
                    
                    # Predicted region — segs are already in feature-space
                    pred_seg = segs[best_idx]  # feature-space coordinates
                    pred_start_idx = max(0, int(pred_seg[0].item()))
                    pred_end_idx = min(T_vid, int(pred_seg[1].item()) + 1)
                    if pred_start_idx >= pred_end_idx:
                        pred_start_idx = max(0, pred_end_idx - 1)
                    
                    pred_attn = attn_avg[pred_start_idx:pred_end_idx].mean(dim=0)
                    
                    # Rest of metrics unchanged
                    attn_similarity = F.cosine_similarity(
                        gt_attn.unsqueeze(0), pred_attn.unsqueeze(0)
                    ).item()
                    
                    gt_top_token = gt_attn.argmax().item()
                    pred_top_token = pred_attn.argmax().item()
                    same_top_token = (gt_top_token == pred_top_token)
                    
                    gt_top_weight = gt_attn.max().item()
                    pred_top_weight = pred_attn.max().item()
                    
                    gt_entropy = -(gt_attn * (gt_attn + 1e-8).log()).sum().item()
                    
                    records.append({
                        'tiou': tiou,
                        'layer': layer_idx,
                        'attn_similarity': attn_similarity,
                        'same_top_token': same_top_token,
                        'gt_top_token': gt_top_token,
                        'pred_top_token': pred_top_token,
                        'gt_top_weight': gt_top_weight,
                        'pred_top_weight': pred_top_weight,
                        'gt_entropy': gt_entropy,
                        'success': tiou >= 0.5,
                        'failure': tiou < 0.1,
                    })
            
            self.itr += 1
        
        # ====== Analysis ======
        self._analyze_token_attention(records)


    def _analyze_token_attention(self, records):
        print0("\n" + "=" * 70)
        print0("TOKEN-LEVEL ATTENTION ANALYSIS")
        print0("=" * 70)
        
        for layer in range(self.n_decoder_layers):
            layer_records = [r for r in records if r['layer'] == layer]
            
            successes = [r for r in layer_records if r['success']]
            failures = [r for r in layer_records if r['failure']]
            
            print0(f"\n--- Layer {layer} ---")
            print0(f"Samples: {len(layer_records)} "
                f"(success={len(successes)}, failure={len(failures)})")
            
            # 1. Attention similarity between GT and predicted regions
            sim_success = np.mean([r['attn_similarity'] for r in successes])
            sim_failure = np.mean([r['attn_similarity'] for r in failures])
            
            print0(f"\nAttention similarity (GT region vs predicted region):")
            print0(f"  Success: {sim_success:.4f}")
            print0(f"  Failure: {sim_failure:.4f}")
            print0(f"  Gap:     {sim_success - sim_failure:+.4f}")
            
            # 2. Do they attend to the same top token?
            same_success = np.mean([r['same_top_token'] for r in successes])
            same_failure = np.mean([r['same_top_token'] for r in failures])
            
            print0(f"\nSame top token (GT vs predicted):")
            print0(f"  Success: {same_success*100:.1f}%")
            print0(f"  Failure: {same_failure*100:.1f}%")
            
            # 3. Attention concentration
            gt_weight_success = np.mean([r['gt_top_weight'] for r in successes])
            gt_weight_failure = np.mean([r['gt_top_weight'] for r in failures])
            
            print0(f"\nTop token weight at GT region:")
            print0(f"  Success: {gt_weight_success:.4f}")
            print0(f"  Failure: {gt_weight_failure:.4f}")
            
            # 4. Key diagnostic: attention similarity correlation with tIoU
            sims = np.array([r['attn_similarity'] for r in layer_records])
            tious = np.array([r['tiou'] for r in layer_records])
            corr = np.corrcoef(sims, tious)[0, 1]
            
            print0(f"\nAttention similarity → tIoU correlation: {corr:+.4f}"
                f"  {'*** STRONG' if abs(corr) > 0.2 else '  weak/none'}")
        
        print0("\n" + "=" * 70)
        print0("INTERPRETATION GUIDE")
        print0("=" * 70)
        print0("""
        IF attention similarity is HIGH for both success and failure:
        → All video regions attend to the same text tokens
        → The fusion produces similar features everywhere
        → Cls head can't distinguish → FIX THE CLS HEAD
        → Multi-hop reasoning won't help (attention is already consistent)
        
        IF attention similarity is LOW for failures but HIGH for success:
        → Failed predictions attend to DIFFERENT tokens than GT
        → The fusion is giving wrong features to wrong regions  
        → Cls head never had a chance → FIX THE FUSION
        → Multi-hop reasoning WOULD help (different hops, different tokens)
        
        IF same_top_token is similar for success and failure:
        → The model always focuses on the same token (likely a noun)
        → It ignores action words that distinguish moments
        → Multi-hop would help: hop 1 finds the object, hop 2 finds the action
        """)

    @torch.no_grad()
    def run_failure_analysis(self):
        """Analyze what input properties predict success/failure."""
        print0("Failure analysis started.")
        
        records = []
        
        for data_list in self.dataloader:
            data = data_list[0]
            results = self.predict(data)
            targets = data['segment']
            
            tokens = data['text']
            if not isinstance(tokens, tuple):
                tokens = (tokens,)
            
            duration = data['duration']
            vid_len = data['vid'].size(-1)
            
            for q_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                
                gt_start, gt_end = target[0], target[1]
                gt_duration = gt_end - gt_start
                
                if len(segs) == 0:
                    tiou = 0.0
                    pred_start, pred_end = 0.0, 0.0
                    center_in_gt = False
                    gt_coverage = 0.0
                    pred_duration_ratio = 0.0
                else:
                    best_idx = scores.argmax()
                    best_seg = segs[best_idx]
                    pred_start = best_seg[0].item()
                    pred_end = best_seg[1].item()
                    
                    target_t = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
                    tiou = iou(best_seg.unsqueeze(0), target_t).item()
                    
                    # Is predicted center within ground truth?
                    pred_center = (pred_start + pred_end) / 2
                    center_in_gt = (pred_center >= gt_start) and (pred_center <= gt_end)
                    
                    # How much of GT does the prediction cover?
                    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
                    gt_coverage = intersection / (gt_duration + 1e-8)
                    
                    # Prediction length vs GT length
                    pred_dur = pred_end - pred_start
                    pred_duration_ratio = pred_dur / (gt_duration + 1e-8)
                
                # Query properties
                query_tokens = tokens[q_idx] if len(tokens) > 1 else tokens[0]
                query_length = (query_tokens != 0).sum().item() if query_tokens.dim() == 1 \
                            else query_tokens.size(-1)
                
                records.append({
                    'tiou': tiou,
                    # Input properties
                    'query_length': query_length,
                    'moment_duration': gt_duration,
                    'moment_coverage': gt_duration / duration,
                    'moment_position': (gt_start + gt_end) / (2 * duration),
                    'video_duration': duration,
                    # Prediction quality breakdown
                    'center_in_gt': center_in_gt,
                    'gt_coverage': gt_coverage,
                    'pred_duration_ratio': pred_duration_ratio,
                    # Raw values for detailed analysis
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                })
            
            self.itr += 1
        
        self._analyze_failures(records)


    def _analyze_failures(self, records):
        import matplotlib.pyplot as plt
        
        save_dir = os.path.join(self.opt['_root'], 'failure_analysis')
        os.makedirs(save_dir, exist_ok=True)
        
        tious = np.array([r['tiou'] for r in records])
        
        # ====== Part 1: What input properties correlate with failure? ======
        print0("\n" + "=" * 70)
        print0("PART 1: INPUT PROPERTY CORRELATIONS")
        print0("=" * 70)
        
        input_props = ['query_length', 'moment_duration', 'moment_coverage', 
                    'moment_position', 'video_duration']
        
        for prop in input_props:
            values = np.array([r[prop] for r in records])
            corr = np.corrcoef(values, tious)[0, 1]
            
            # Also compute binned performance
            bins = np.percentile(values, [0, 25, 50, 75, 100])
            bin_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
            bin_tious = []
            for b in range(4):
                if b < 3:
                    mask = (values >= bins[b]) & (values < bins[b + 1])
                else:
                    mask = (values >= bins[b]) & (values <= bins[b + 1])
                if mask.sum() > 0:
                    bin_tious.append(tious[mask].mean())
                else:
                    bin_tious.append(0)
            
            trend = "↗" if bin_tious[-1] > bin_tious[0] + 0.02 else \
                    "↘" if bin_tious[-1] < bin_tious[0] - 0.02 else "→"
            
            strength = "*** STRONG" if abs(corr) > 0.3 else \
                    "**  MODERATE" if abs(corr) > 0.15 else \
                    "*   WEAK" if abs(corr) > 0.08 else "    NONE"
            
            print0(f"\n{prop:20s}  r={corr:+.4f}  {strength}  {trend}")
            print0(f"  Q1={bin_tious[0]:.3f}  Q2={bin_tious[1]:.3f}  "
                f"Q3={bin_tious[2]:.3f}  Q4={bin_tious[3]:.3f}")
            print0(f"  range: [{values.min():.2f}, {values.max():.2f}], "
                f"mean={values.mean():.2f}")
        
        # ====== Part 2: Where do predictions fail? ======
        print0("\n" + "=" * 70)
        print0("PART 2: FAILURE MODE ANALYSIS")
        print0("=" * 70)
        
        center_correct = np.array([r['center_in_gt'] for r in records])
        gt_coverage = np.array([r['gt_coverage'] for r in records])
        pred_ratio = np.array([r['pred_duration_ratio'] for r in records])
        
        # Categorize failures
        success = tious >= 0.5  # IoU >= 0.5
        partial = (tious >= 0.1) & (tious < 0.5)
        failure = tious < 0.1
        
        print0(f"\nOverall: {len(records)} samples")
        print0(f"  Success (IoU≥0.5): {success.sum()} ({success.mean()*100:.1f}%)")
        print0(f"  Partial (0.1≤IoU<0.5): {partial.sum()} ({partial.mean()*100:.1f}%)")
        print0(f"  Failure (IoU<0.1): {failure.sum()} ({failure.mean()*100:.1f}%)")
        
        print0(f"\nCenter-in-GT accuracy:")
        print0(f"  Overall:  {center_correct.mean()*100:.1f}%")
        print0(f"  Success:  {center_correct[success].mean()*100:.1f}%")
        print0(f"  Partial:  {center_correct[partial].mean()*100:.1f}%")
        print0(f"  Failure:  {center_correct[failure].mean()*100:.1f}%")
        
        print0(f"\nGT coverage (how much of GT the prediction overlaps):")
        print0(f"  Overall:  {gt_coverage.mean():.3f}")
        print0(f"  Success:  {gt_coverage[success].mean():.3f}")
        print0(f"  Partial:  {gt_coverage[partial].mean():.3f}")
        print0(f"  Failure:  {gt_coverage[failure].mean():.3f}")
        
        print0(f"\nPrediction duration ratio (pred_length / gt_length):")
        print0(f"  Overall:  {pred_ratio.mean():.3f}")
        print0(f"  Success:  {pred_ratio[success].mean():.3f}")
        print0(f"  Partial:  {pred_ratio[partial].mean():.3f}")
        print0(f"  Failure:  {pred_ratio[failure].mean():.3f}")
        
        # Failure subcategories
        if partial.sum() > 0:
            partial_records = [r for r, p in zip(records, partial) if p]
            
            # Right region, wrong boundaries
            right_region = sum(1 for r in partial_records if r['center_in_gt'])
            # Wrong region entirely
            wrong_region = sum(1 for r in partial_records if not r['center_in_gt'])
            # Too short predictions
            too_short = sum(1 for r in partial_records 
                            if r['pred_duration_ratio'] < 0.5 and r['center_in_gt'])
            # Too long predictions
            too_long = sum(1 for r in partial_records 
                        if r['pred_duration_ratio'] > 2.0 and r['center_in_gt'])
            # Offset (right length, wrong position)
            offset = right_region - too_short - too_long
            
            total_partial = len(partial_records)
            print0(f"\nPartial failure breakdown ({total_partial} samples):")
            print0(f"  Right region, wrong boundaries: {right_region} ({right_region/total_partial*100:.1f}%)")
            print0(f"    - Too short (ratio<0.5): {too_short} ({too_short/total_partial*100:.1f}%)")
            print0(f"    - Too long  (ratio>2.0): {too_long} ({too_long/total_partial*100:.1f}%)")
            print0(f"    - Offset (right length): {offset} ({offset/total_partial*100:.1f}%)")
            print0(f"  Wrong region entirely:          {wrong_region} ({wrong_region/total_partial*100:.1f}%)")
        
        # ====== Part 3: Visualizations ======
        
        # Plot: input properties vs tIoU
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, prop in enumerate(input_props):
            values = np.array([r[prop] for r in records])
            ax = axes[idx]
            
            # Binned bar chart
            n_bins = 8
            bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            
            bin_means, bin_centers = [], []
            for b in range(len(bins) - 1):
                if b < len(bins) - 2:
                    mask = (values >= bins[b]) & (values < bins[b + 1])
                else:
                    mask = (values >= bins[b]) & (values <= bins[b + 1])
                if mask.sum() > 0:
                    bin_means.append(tious[mask].mean())
                    bin_centers.append((bins[b] + bins[b + 1]) / 2)
            
            ax.bar(range(len(bin_means)), bin_means, color='steelblue', 
                edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(bin_means)))
            ax.set_xticklabels([f'{c:.1f}' for c in bin_centers], fontsize=8)
            ax.set_ylabel('Mean tIoU')
            ax.set_xlabel(prop)
            
            corr = np.corrcoef(values, tious)[0, 1]
            ax.set_title(f'{prop}\nr={corr:+.3f}')
        
        axes[-1].set_visible(False)
        plt.suptitle('Input Properties vs Prediction Quality', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'input_vs_tiou.png'), dpi=150)
        plt.close()
        
        # Plot: failure mode pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.pie([success.sum(), partial.sum(), failure.sum()],
                labels=['Success\n(IoU≥0.5)', 'Partial\n(0.1≤IoU<0.5)', 'Failure\n(IoU<0.1)'],
                colors=['#2ecc71', '#f39c12', '#e74c3c'],
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Performance')
        
        # Prediction length ratio distribution
        ax2.hist(pred_ratio[pred_ratio < 5], bins=50, 
                color='steelblue', edgecolor='black', linewidth=0.5)
        ax2.axvline(x=1.0, color='red', linestyle='--', label='Perfect ratio')
        ax2.set_xlabel('Predicted Duration / GT Duration')
        ax2.set_ylabel('Count')
        ax2.set_title('Duration Ratio Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'failure_modes.png'), dpi=150)
        plt.close()
        
        print0(f"\nPlots saved to {save_dir}")
    # Add this to your failure analysis
    @torch.no_grad()
    def run_confidence_analysis(self):
        """Check if prediction confidence predicts correctness."""
        
        records = []
        
        for data_list in self.dataloader:
            data = data_list[0]
            results = self.predict(data)
            targets = data['segment']
            
            for q_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                if len(segs) == 0:
                    continue
                
                # Top-1 prediction
                sorted_idx = scores.argsort(descending=True)
                top1_score = scores[sorted_idx[0]].item()
                top1_seg = segs[sorted_idx[0]]
                
                target_t = torch.as_tensor(target, dtype=torch.float).unsqueeze(0)
                top1_tiou = iou(top1_seg.unsqueeze(0), target_t).item()
                
                # Score gap between top-1 and top-2
                if len(scores) > 1:
                    top2_score = scores[sorted_idx[1]].item()
                    score_gap = top1_score - top2_score
                    score_ratio = top1_score / (top2_score + 1e-8)
                else:
                    score_gap = top1_score
                    score_ratio = float('inf')
                
                # Top-5 score statistics
                top5_scores = scores[sorted_idx[:min(5, len(scores))]].numpy()
                score_entropy = -(top5_scores * np.log(top5_scores + 1e-8)).sum()
                
                # Check if any of top-5 would have been correct
                top5_tious = []
                for k in range(min(5, len(segs))):
                    k_tiou = iou(segs[sorted_idx[k]].unsqueeze(0), target_t).item()
                    top5_tious.append(k_tiou)
                
                best_in_top5 = max(top5_tious)
                correct_rank = None
                for rank, t in enumerate(top5_tious):
                    if t >= 0.5:
                        correct_rank = rank + 1
                        break
                
                records.append({
                    'top1_tiou': top1_tiou,
                    'top1_score': top1_score,
                    'score_gap': score_gap,
                    'score_ratio': score_ratio,
                    'score_entropy': score_entropy,
                    'best_in_top5': best_in_top5,
                    'correct_rank': correct_rank,  # None if not in top-5
                    'top1_correct': top1_tiou >= 0.5,
                    'recoverable': best_in_top5 >= 0.5 and top1_tiou < 0.5,
                })
            
            self.itr += 1
        
        # ====== Analysis ======
        print0("\n" + "=" * 70)
        print0("CONFIDENCE & RE-RANKING ANALYSIS")
        print0("=" * 70)
        
        top1_tious = np.array([r['top1_tiou'] for r in records])
        top1_scores = np.array([r['top1_score'] for r in records])
        score_gaps = np.array([r['score_gap'] for r in records])
        
        # 1. Does confidence predict correctness?
        corr = np.corrcoef(top1_scores, top1_tious)[0, 1]
        print0(f"\nTop-1 score → tIoU correlation: {corr:+.4f}")
        
        corr_gap = np.corrcoef(score_gaps, top1_tious)[0, 1]
        print0(f"Score gap (top1-top2) → tIoU correlation: {corr_gap:+.4f}")
        
        # 2. How many failures are RECOVERABLE by re-ranking?
        top1_correct = sum(1 for r in records if r['top1_correct'])
        recoverable = sum(1 for r in records if r['recoverable'])
        total = len(records)
        
        print0(f"\nTop-1 accuracy (IoU≥0.5): {top1_correct}/{total} "
            f"({top1_correct/total*100:.1f}%)")
        print0(f"Recoverable by re-ranking top-5: {recoverable}/{total} "
            f"({recoverable/total*100:.1f}%)")
        print0(f"Potential accuracy with oracle top-5: "
            f"{(top1_correct + recoverable)/total*100:.1f}%")
        
        # 3. Where are correct predictions ranked?
        rank_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'not_in_top5': 0}
        for r in records:
            if r['correct_rank'] is not None:
                rank_dist[r['correct_rank']] += 1
            else:
                rank_dist['not_in_top5'] += 1
        
        print0(f"\nCorrect prediction rank distribution:")
        for rank, count in rank_dist.items():
            bar = '█' * int(count / total * 100)
            print0(f"  Rank {rank:>12}: {count:>5} ({count/total*100:5.1f}%) {bar}")
        
        # 4. Confidence binned analysis
        print0(f"\nConfidence-binned accuracy:")
        bins = np.percentile(top1_scores, [0, 25, 50, 75, 100])
        for b in range(4):
            if b < 3:
                mask = (top1_scores >= bins[b]) & (top1_scores < bins[b + 1])
            else:
                mask = (top1_scores >= bins[b]) & (top1_scores <= bins[b + 1])
            
            if mask.sum() > 0:
                acc = (top1_tious[mask] >= 0.5).mean()
                mean_tiou = top1_tious[mask].mean()
                print0(f"  Score Q{b+1} [{bins[b]:.3f}-{bins[b+1]:.3f}]: "
                    f"acc={acc*100:.1f}%, mean_tIoU={mean_tiou:.3f}, "
                    f"n={mask.sum()}")
        
        print0("=" * 70)

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)