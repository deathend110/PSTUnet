import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class TVLoss(nn.Module):
    """
    全变分损失 (Total Variation Loss)
    用于平滑 1-bit 量化带来的背景高频杂乱散焦伪影
    """
    def __init__(self, TVLoss_weight=1e-3):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class SSIMLoss(nn.Module):
    """
    可微的 SSIM 损失函数 (PyTorch 原生实现)
    这里保留它，只用于验证阶段统计 SSIM，不参与训练 loss。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class Seq_SAR_L1TVLoss(nn.Module):
    """
    专门处理 5D 张量 [B, C, T, H, W] 的纯 L1 + TV 损失
    """
    def __init__(self, tv_weight=1e-3):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss(TVLoss_weight=tv_weight)

    def forward(self, pred, target):
        B, C, T, H, W = pred.shape

        pred_2d = pred.transpose(1, 2).contiguous().view(B * T, C, H, W)
        target_2d = target.transpose(1, 2).contiguous().view(B * T, C, H, W)

        loss_l1 = self.l1_loss(pred_2d, target_2d)
        loss_tv = self.tv_loss(pred_2d)

        return loss_l1 + loss_tv


# 为了兼容旧代码里可能还在 import Seq_SAR_HybridLoss
# 这里保留一个同名别名，但实际内容仍然是纯 L1 + TV
class Seq_SAR_HybridLoss(Seq_SAR_L1TVLoss):
    def __init__(self, tv_weight=1e-3, ssim_weight=0.0):
        super().__init__(tv_weight=tv_weight)
        self.ssim_weight = 0.0


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        import math

        if math.isnan(val_loss) or math.isinf(val_loss):
            self.counter += 1
            print(f'🚨 EarlyStopping 检测到 NaN！强制警告器 +1: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss