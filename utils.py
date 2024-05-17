import random
import numpy as np
import torch
from torch import nn
from model.WaveNet import WaveNet
from model.loss_fn import DiceLoss


def get_model(args):
    model = WaveNet(class_num=args.class_num,
                    number_levels=args.num_blocks,
                    reconstruction=None,
                    w_act=args.w_act,
                    gate_pool=args.gate_pool,
                    reg_details=args.reg_details,
                    reg_approx=args.reg_approx,
                    w_position=args.w_position,
                    block_type=args.block_type,
                    haar_wavelet=args.haar,
                    d_model=args.d_model,
                    num_trans_layers=args.num_trans_layers)

    return model


def create_loss_fn(args):
    print('Loss weights: ', args.loss_weights)
    if args.loss_name == 'BCE':
        return nn.BCELoss(weight=args.loss_weights)
    if args.loss_name == 'Dice':
        return DiceLoss(weights=args.loss_weights)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
