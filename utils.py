import os
import time
import shutil
import torch.nn as nn
import einops
import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import random
import torch.nn.functional as F
import torch



def random_selection(input_list):
    num_to_select = random.randint(1, 3)  # 随机选择1到3个数字
    selected_numbers = random.sample(input_list, num_to_select)  # 从列表中选择随机数字
    return selected_numbers

class Loss_CC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, m):
        b, c, h, w = m.shape
        m = m.reshape(b, c, h*w)
        m = torch.nn.functional.normalize(m, dim=2, p=2)
        m_T = torch.transpose(m, 1, 2)
        m_cc = torch.matmul(m, m_T)
        mask = torch.eye(c).unsqueeze(0).repeat(b,1,1).cuda()
        m_cc = m_cc.masked_fill(mask==1, 0)
        loss = torch.sum(m_cc**2)/(b*c*(c-1))
        return loss
    
class norm_mse_loss(nn.Module):
    def __init__(self, patch_size=8) -> None:
        super().__init__()
        self.eps = 1e-4
        self.patch_size = patch_size
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        h_size = pred.shape[2] // self.patch_size
        w_size = pred.shape[3] // self.patch_size
        pred = einops.rearrange(pred,' b c (h p1) (w p2) -> b (h w) (c p1 p2)',h=h_size,w=w_size,p1=self.patch_size,p2=self.patch_size)
        target = einops.rearrange(target,'b c (h p1) (w p2) -> b (h w) (c p1 p2)',h=h_size,w=w_size,p1=self.patch_size,p2=self.patch_size)
        normed_pred = (pred - torch.mean(pred, dim=-1, keepdim=True)) / (torch.std(pred, dim=-1, keepdim=True) + self.eps)
        normed_target = (target - torch.mean(target, dim=-1, keepdim=True)) / (torch.std(target, dim=-1, keepdim=True) + self.eps)
        norm_l2 = F.mse_loss(normed_pred, normed_target,reduction='none').mean()
        return norm_l2

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse)
