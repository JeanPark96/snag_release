import argparse
import os
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from libs import load_opt
from libs import TrainerDecGate as Trainer


def main(rank, opt):
    torch.cuda.set_device(rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(f"Training process: {rank}")
    
    if opt['_distributed']:
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=opt['_world_size']
        )
    trainer = Trainer(opt)
    trainer.run()
    if opt['_distributed']:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help="training options")
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--folder', type=str, default="base", help="experiment name")
    parser.add_argument('--seed', type=int, default=1234567891, help="overwrite seed")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = False

    # create experiment folder
    os.makedirs('experiments', exist_ok=True)
    root = os.path.join('experiments', args.folder, args.name)
    os.makedirs(root, exist_ok=True)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'), is_training=True)
    except:
        opt_path = os.path.join('opts', args.opt)
        opt = load_opt(opt_path, is_training=True)
        shutil.copyfile(opt_path, os.path.join(root, 'opt.yaml'))
        os.makedirs(os.path.join(root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(root, 'states'), exist_ok=True)
    print("original seed", opt["seed"], " overwrite with ", args.seed)
    opt["seed"] = args.seed
    opt['_root'] = root
    opt['_resume'] = (
        os.path.exists(os.path.join(root, 'models', 'last.pth'))
        and os.path.exists(os.path.join(root, 'states', 'last.pth'))
    )
    #n_gpus = 1
    n_gpus = torch.cuda.device_count()
    # set up distributed training
    ## NOTE: only supports single-node training
    opt['_world_size'] = n_gpus
    opt['_distributed'] = n_gpus > 1
    if opt['_distributed']:
        mp.spawn(main, nprocs=n_gpus, args=(opt, ))
    else:
        main(0, opt)