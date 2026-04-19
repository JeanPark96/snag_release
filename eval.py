import argparse
import os

import torch
from libs import load_opt
from libs import EvaluatorWithLog as Evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ckpt', type=str, help="checkpoint name")
    parser.add_argument('--folder', type=str, default="base", help="experiment name")
    parser.add_argument('--split', type=str, default=None,
                    help='Override eval split (train/val/test)')
    parser.add_argument('--return_features', action='store_true',
                    help='Whether to return intermediate features for analysis')
    args = parser.parse_args()
    experiment_folder_name = "experiments_train_val_split"
    root = os.path.join(experiment_folder_name, args.folder, args.name)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'), is_training=False)
    except:
        raise ValueError('experiment folder not found')
    assert os.path.exists(os.path.join(root, 'models', f'{args.ckpt}.pth'))
    opt['_root'] = root
    opt['_ckpt'] = args.ckpt

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.split is not None:
        opt['eval']['data']['split'] = args.split
    evaluator = Evaluator(opt, return_features=args.return_features)
    evaluator.run()
    # evaluator.run_failure_analysis()
    # evaluator.run_confidence_analysis()
    #evaluator.run_token_attention_analysis()