"""
Trainer with validation support.

Changes from original:
  1. Loads a validation dataset/dataloader alongside training data.
  2. Runs validation (forward-only on EMA model) at the end of each epoch.
  3. Tracks best validation loss and saves the EMA checkpoint as best.
  4. Optionally runs Evaluator on val set to display R@IoU metrics.

Flow per epoch:
  1. Train (update raw model weights)
  2. EMA update (after each step)
  3. Validate using EMA model: compute loss + R@IoU (display only)
  4. If best val loss → save EMA state dict as best.pth

Saved checkpoints:
  - last.pth: raw model + EMA model (every epoch, for resuming)
  - best.pth: EMA model only (best val loss)

Config additions under opt['train']:
    val_data:           dict, same schema as opt['train']['data'] but for val split
                        (use is_training=True, crop_ratio=null)
    val_batch_size:     int, batch size for validation (default: same as train)
    val_interval:       int, validate every N epochs (default: 1)
    val_eval:           dict, same schema as opt['eval'] but pointing to val split
                        (optional, for R@IoU display)
"""


from collections import OrderedDict
from copy import deepcopy
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, PtTransformer,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str


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
 
        # ── training dataset ──
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
 
        # ── validation dataset ──
        val_data_opt = opt['train'].get('val_data')
        if val_data_opt is not None:
            # merge: train.data as base, val_data overrides (e.g. split)
            merged_val_data = {**deepcopy(opt['train']['data']), **val_data_opt}
            self.val_dataset = make_dataset(
                merged_val_data, num_epochs=1, is_training=True
                # is_training=True so we get targets in the same format
            )
            val_batch_size = opt['train'].get('val_batch_size', batch_size)
            self.val_batch_size = val_batch_size
            # use a separate rng so val shuffling doesn't disturb train
            val_rng = fix_random_seed(opt.get('seed', 2022) + 7777)
            self.val_dataloader, self.val_sampler = make_dataloader(
                self.val_dataset, generator=val_rng, is_training=False,
                batch_size=val_batch_size,
                num_workers=opt['train'].get('val_num_workers',
                                             opt['train']['num_workers']),
                world_size=get_world_size(), rank=get_rank()
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None
            self.val_sampler = None
            self.val_batch_size = None
 
        self.val_interval = opt['train'].get('val_interval', 1)
 
        # ── val evaluator for R@IoU display ──
        val_eval_cfg = opt['train'].get('val_eval')
        if val_eval_cfg is not None:
            # merge val_eval.data with train.data as base
            merged_eval_data = {
                **deepcopy(opt['train']['data']),
                **val_eval_cfg.get('data', {})
            }
            eval_opt = deepcopy(opt)
            eval_opt['eval'] = deepcopy(val_eval_cfg)
            eval_opt['eval']['data'] = merged_eval_data
            self.val_evaluator = Evaluator(eval_opt, model=self.model_ema)
        else:
            self.val_evaluator = None
 
        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')
 
        # best validation tracking
        self.best_val_loss = float('inf')
 
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
            self.model = DistributedDataParallel(
                self.model, [get_rank()], find_unused_parameters=False
            )
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
 
    # ──────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────
 
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
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
 
            # ── validation ──
            if (self.val_dataloader is not None
                    and self.epoch % self.val_interval == 0):
                val_loss = self.validate()
                if get_rank() == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_best()
 
            self.checkpoint()
            barrier()
        print0("Training completed.")
 
    # ──────────────────────────────────────────────────────────────────
    # Training forward / backward
    # ──────────────────────────────────────────────────────────────────
 
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
 
        fpn_logits = torch.cat(fpn_logits, dim=1)
        fpn_offsets = torch.cat(fpn_offsets, dim=1)
        fpn_masks = torch.cat(fpn_masks, dim=1)
        points = torch.cat(fpn_points)
 
        gt_labels, gt_offsets = self._annotate_points(points, targets)
 
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()
 
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
 
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
 
    # ──────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def validate(self):
        """Run forward pass on the validation set using EMA model.
        Returns average total loss."""
        self.model_ema.eval()
 
        # ── (1) val loss ──
        val_cls = val_reg = val_total = val_count = 0
 
        for data_list in self.val_dataloader:
            loss_dict = self._val_forward(data_list, self.model_ema)
            bs = len(data_list)
            val_cls += loss_dict['cls'].item() * bs
            val_reg += loss_dict['reg'].item() * bs
            val_total += loss_dict['total'].item() * bs
            val_count += bs
 
        if val_count == 0:
            return float('inf')
 
        avg_cls = val_cls / val_count
        avg_reg = val_reg / val_count
        avg_total = val_total / val_count
 
        if get_rank() == 0:
            self.tb_writer.add_scalar('val_ema/cls', avg_cls, self.epoch)
            self.tb_writer.add_scalar('val_ema/reg', avg_reg, self.epoch)
            self.tb_writer.add_scalar('val_ema/total', avg_total, self.epoch)
            self.logger.write(
                f"[Val epoch {self.epoch}] (EMA) "
                f"cls {avg_cls:.4f} | reg {avg_reg:.4f} | total {avg_total:.4f}"
            )
            if avg_total < self.best_val_loss:
                self.logger.write(
                    f"  *** New best val loss: {avg_total:.4f} "
                    f"(prev {self.best_val_loss:.4f})"
                )
 
        # ── (2) R@IoU (display only) ──
        if self.val_evaluator is not None:
            metrics = self.val_evaluator.evaluate(self.model_ema)
            if get_rank() == 0:
                parts = []
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(f'val_ema/{k}', v, self.epoch)
                    parts.append(f"{k} {v:.2f}")
                self.tb_writer.flush()
                self.logger.write(
                    f"[Val epoch {self.epoch}] (EMA) " + " | ".join(parts)
                )
 
        return avg_total
 
    def _val_forward(self, data_list, eval_model):
        """Forward-only pass on a single batch for validation."""
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
 
        fpn_logits, fpn_offsets, fpn_masks, _ = \
            eval_model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)
 
        fpn_logits = torch.cat(fpn_logits, dim=1)
        fpn_offsets = torch.cat(fpn_offsets, dim=1)
        fpn_masks = torch.cat(fpn_masks, dim=1)
        points = torch.cat(fpn_points)
 
        gt_labels, gt_offsets = self._annotate_points(points, targets)
 
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = max(pos_masks.sum().item(), 1)
 
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / norm
 
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / norm
 
        total_loss = cls_loss + self.loss_weight * reg_loss
 
        return {
            'cls': total_loss.new_tensor(cls_loss.item()),
            'reg': total_loss.new_tensor(reg_loss.item()),
            'total': total_loss.new_tensor(total_loss.item()),
        }
 
    # ──────────────────────────────────────────────────────────────────
    # Batching helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────
 
    def _batchify_videos(self, vid_list):
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
 
        vid, vid_masks = self._batchify_videos(vid_list)
 
        if isinstance(text_list[0], tuple):
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)
 
        text_size = torch.as_tensor(n)
        return vid, vid_masks, text, text_masks, text_size
 
    # ──────────────────────────────────────────────────────────────────
    # Annotation / loss helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────
 
    def _annotate_points(self, points, targets):
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets
 
    def _annotate_points_per_video(self, points, target):
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]
 
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
 
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )
        labels = torch.logical_and(inside_window, inside_range)
        return labels, offsets
 
    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')
 
    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')
 
    # ──────────────────────────────────────────────────────────────────
    # EMA helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────
 
    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())
 
    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))
 
    # ──────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────
 
    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model
 
    def _save_best(self):
        """Save best EMA checkpoint (rank 0 only)."""
        model_dir = os.path.join(self.opt['_root'], 'models')
        os.makedirs(model_dir, exist_ok=True)
        best_ckpt = {
            'best_ema_state_dict': self.model_ema.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.epoch,
        }
        torch.save(best_ckpt, os.path.join(model_dir, 'best.pth'))
        print0(f"Saved best EMA model at epoch {self.epoch} "
               f"(val_loss={self.best_val_loss:.4f})")
 
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
        # restore best val loss from state or from best.pth
        if 'best_val_loss' in state_ckpt:
            self.best_val_loss = state_ckpt['best_val_loss']
        else:
            best_path = os.path.join(self.opt['_root'], 'models', 'best.pth')
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location='cpu')
                self.best_val_loss = best_ckpt.get('best_val_loss', float('inf'))
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
 
    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )
 
    # ──────────────────────────────────────────────────────────────────
    # Logging (unchanged)
    # ──────────────────────────────────────────────────────────────────
 
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

class Evaluator:

    def __init__(self, opt, model=None):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)

        # model
        if model is not None:
            # external model (e.g. EMA from Trainer)
            self.model = model
            self._external_model = True
        else:
            # standalone: load from checkpoint
            self.model = PtTransformer(opt['model']).cuda()
            self.load_model()
            self.model.eval().requires_grad_(False)
            self._external_model = False

        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # logging (standalone mode only)
        self.logger = None
        if not self._external_model:
            self.logger = Logger(
                os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt")
            )

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

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, model=None):
        """Run evaluation and return metrics dict.

        Args:
            model: optional model to use (overrides self.model for this call).

        Returns:
            dict with keys like 'R1@0.3', 'R1@0.5', 'R5@0.5', etc.
        """
        eval_model = model if model is not None else self.model
        was_training = eval_model.training
        eval_model.eval()

        counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        text_cnt = 0

        for data_list in self.dataloader:
            results = self._predict(data_list[0], eval_model)
            targets = data_list[0]['segment']

            assert len(results) == len(targets)
            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs = segs[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)

                iou_topk = iou(segs, target)
                iou_n = np.array(
                    [iou_topk[:r].max().item() if r <= len(iou_topk) else 0.0
                     for r in self.ranks]
                )
                counts += (iou_n[:, None] >= self.iou_threshs[None])
            text_cnt += len(targets)

        if was_training:
            eval_model.train()

        # assemble metrics dict
        metrics_mat = counts / max(text_cnt, 1)
        metrics = {}
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                metrics[f'R{rank}@{thresh:.1f}'] = metrics_mat[i, j] * 100
        return metrics

    @torch.no_grad()
    def run(self):
        """Standalone evaluation with logging (original behavior)."""
        print0("Evaluation started.")
        start_time = time.time()

        counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        text_cnt = 0
        log_interval = self.num_itrs // 10
        itr = 0

        for data_list in self.dataloader:
            results = self._predict(data_list[0], self.model)
            targets = data_list[0]['segment']

            assert len(results) == len(targets)
            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs = segs[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)

                iou_topk = iou(segs, target)
                iou_n = np.array(
                    [iou_topk[:r].max().item() if r <= len(iou_topk) else 0.0
                     for r in self.ranks]
                )
                counts += (iou_n[:, None] >= self.iou_threshs[None])
            text_cnt += len(targets)
            itr += 1

            if self.logger and (itr == 1 or itr % log_interval == 0):
                self._log(counts, text_cnt, itr, self.num_itrs)

        if self.logger:
            self._log(counts, text_cnt, itr, self.num_itrs, is_last=True)
        print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

        metrics_mat = counts / max(text_cnt, 1)
        metrics = {}
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                metrics[f'R{rank}@{thresh:.1f}'] = metrics_mat[i, j] * 100
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    def _predict(self, data, model):
        """Predict event segments given a single video and text queries."""
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

            text, text_mask = model.encode_text(text, text_mask)
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

            fpn, fpn_masks = model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for text, text_mask in zip(text_list, text_mask_list):
                fpn_logits, fpn_offsets, _, _ = \
                    model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            window_segs_list, window_scores_list = tuple(), tuple()
            for q_idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks,
                    window_ext[q_idx] if window_ext is not None else None
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
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
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

    def _log(self, counts, text_cnt, itr, num_itrs, is_last=False):
        metrics = counts / max(text_cnt, 1)
        log_str = "\nFinal:" if is_last else f"\n[{itr}/{num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)