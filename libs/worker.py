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
            fpn_logits, fpn_offsets, fpn_masks = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks = \
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
    # def _microbatch_forward_backward(self, data_list, is_last=False):
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

    #     # forward pass — now returns intermediate predictions
    #     if is_last or not self.opt['_distributed']:
    #         fpn_logits, fpn_offsets, fpn_masks, all_logits, all_offsets = \
    #             self.model(vid, vid_masks, text, text_masks, text_size)
    #     else:
    #         with self.model.no_sync():
    #             fpn_logits, fpn_offsets, fpn_masks, all_logits, all_offsets = \
    #                 self.model(vid, vid_masks, text, text_masks, text_size)

    #     fpn_n_points = [m.size(-1) for m in fpn_masks]
    #     fpn_points = self.pt_gen(fpn_n_points)
    #     points = torch.cat(fpn_points)
    #     gt_labels, gt_offsets = self._annotate_points(points, targets)

    #     # === compute loss for ALL iterations ===
    #     total_loss = 0
    #     final_cls_loss = None
    #     final_reg_loss = None

    #     # for iter_idx in range(len(all_logits)):
    #     #     iter_logits = torch.cat(all_logits[iter_idx], dim=1)
    #     #     iter_offsets = torch.cat(all_offsets[iter_idx], dim=1)
    #     #     iter_masks = torch.cat(fpn_masks, dim=1)

    #     #     pos_masks = torch.logical_and(gt_labels, iter_masks)
    #     #     norm = max(pos_masks.sum().item(), 1)

    #     #     cls_loss_i = self._calc_focal_loss(
    #     #         logits=iter_logits[iter_masks],
    #     #         labels=gt_labels[iter_masks]
    #     #     ) / self.loss_norm * get_world_size()

    #     #     reg_loss_i = self._calc_iou_loss(
    #     #         pred_offsets=iter_offsets[pos_masks],
    #     #         gt_offsets=gt_offsets[pos_masks]
    #     #     ) / self.loss_norm * get_world_size()

    #     #     # weight: final iteration = 1.0, intermediate = 0.5
    #     #     is_final = (iter_idx == len(all_logits) - 1)
    #     #     weight = 1.0 if is_final else 0.5

    #     #     total_loss += weight * (cls_loss_i + self.loss_weight * reg_loss_i)

    #     #     if is_final:
    #     #         final_cls_loss = cls_loss_i.detach()
    #     #         final_reg_loss = reg_loss_i.detach()

    #     # total_loss.backward()

    #     iter_logits = torch.cat(all_logits[-1], dim=1)
    #     iter_offsets = torch.cat(all_offsets[-1], dim=1)
    #     iter_masks = torch.cat(fpn_masks, dim=1)

    #     pos_masks = torch.logical_and(gt_labels, iter_masks)

    #     cls_loss = self._calc_focal_loss(
    #         logits=iter_logits[iter_masks],
    #         labels=gt_labels[iter_masks]
    #     ) / self.loss_norm * get_world_size()

    #     reg_loss = self._calc_iou_loss(
    #         pred_offsets=iter_offsets[pos_masks],
    #         gt_offsets=gt_offsets[pos_masks]
    #     ) / self.loss_norm * get_world_size()

    #     total_loss = cls_loss + self.loss_weight * reg_loss
    #     total_loss.backward()

    #     return {
    #         'cls': cls_loss.detach(),
    #         'reg': reg_loss.detach(),
    #         'total': total_loss.detach(),
    #         'norm': pos_masks.sum().detach(),
    #     }


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
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        
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
                fpn_logits, fpn_offsets, _ = \
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
                
                fpn_logits, fpn_offsets, _ = \
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
                fpn_logits, fpn_offsets, _ = \
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
                fpn_logits, fpn_offsets, _ = \
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