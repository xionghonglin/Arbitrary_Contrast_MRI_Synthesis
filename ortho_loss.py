import torch

class OrthoLoss(torch.nn.Module):
    def __init__(self):
        super(OrthoLoss, self).__init__()
    
    def forward(self, m):
        b, c, h, w = m.shape
        m = m.reshape(b, c, h * w)
        m = torch.nn.functional.normalize(m, dim=2, p=2)
        m_T = torch.transpose(m, 1, 2)
        m_cc = torch.matmul(m, m_T)
        mask = torch.eye(c).unsqueeze(0).repeat(b, 1, 1).cuda()
        m_cc = m_cc.masked_fill(mask == 1, 0).cuda()
        loss = torch.sum(m_cc ** 2) / (b * c * (c - 1))
        return loss

if __name__ == '__main__':
    m = torch.randn(2, 3, 4, 4).cuda()
    loss_func = OrthoLoss()
    loss = loss_func(m)
    print(loss)
