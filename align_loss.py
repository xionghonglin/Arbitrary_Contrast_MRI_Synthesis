import torch
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def structural_similarity(img1, img2, window_size=11, window=None, size_average=True):
    (_, channel, height, width) = img1.size()
    
    if window is None or window.size(0) != channel:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device).type(img1.dtype)
    
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - (
        F.conv2d(img1, window, padding=0, groups=channel) * F.conv2d(img2, window, padding=0, groups=channel))
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - F.conv2d(img1, window, padding=0, groups=channel) ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - F.conv2d(img2, window, padding=0, groups=channel) ** 2
    sigma1_sq = torch.clamp(sigma1_sq, min=1e-6)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-6)

    C2 = 1e-4  # 避免除零
    structural_map = sigma12 / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2)
    
    if size_average:
        return structural_map.mean()
    else:
        return structural_map.mean(dim=[1, 2, 3])

class AlignLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(AlignLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = None
        self.channel = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if self.window is None or self.channel != channel:
            self.window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        return 1 - structural_similarity(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)

# if __name__ == '__main__':
#     m_1 = torch.randn(2, 64, 16, 16).cuda()
#     m_2 = torch.randn(2, 64, 16, 16).cuda()
#     loss_func = AlignLoss()
#     loss = loss_func(m_1, m_2)
#     print(loss)
