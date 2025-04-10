import torch
import torch.nn as nn

import torch.nn.functional as F
#import numpy as np
from models.linear import Linear
import models
from models import register
from models.simple_tokenizer import SimpleTokenizer

def tokenize(tokenizer, text, context_length=4):
        tokens = tokenizer.encode(text)
        result = torch.zeros(context_length)
        result[:len(tokens)] = torch.tensor(tokens)
        result = (result - result.mean()) / result.std()
        return result


@register('lccd')
class LCCD(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, text_prompt = False):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        f_dim = self.encoder.out_dim
        self.fusion_net = Linear(in_dim=4*f_dim,out_dim=f_dim,hidden_list=[3*f_dim,2*f_dim])
        self.codemapping_mlp = Linear(in_dim=4, out_dim=f_dim, hidden_list = [f_dim, f_dim, f_dim])
        self.tokenizer = SimpleTokenizer()
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
        if text_prompt:
            tokens = [
            tokenize(self.tokenizer, 'T1'),
            tokenize(self.tokenizer, 'T2w'),
            tokenize(self.tokenizer, 'FLAIR'),
            tokenize(self.tokenizer, 'Gd')
            ]
            self.condition_lib = torch.stack(tokens).float().cuda()
        else:
            self.condition_lib= torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).float().cuda()

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat

    def forward(self, img_src, m_src):
        b,c,h,w = img_src[0].shape

        feats = [None, None, None, None]
        codes = [None, None, None, None]
        feats_pred = [None, None, None, None]
        preds = [None, None, None, None]
        
        for i in range(4):
            codes[i] = self.codemapping_mlp(self.condition_lib[i])
    
            codes[i] = codes[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            # codes[i] = F.normalize(codes[i], dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
            if i in m_src:
                feats[i] = self.gen_feat(img_src[i])/codes[i]
            else:
                feats[i] = torch.zeros(b, self.encoder.out_dim, h, w).cuda()
 
        feats_all = torch.cat(feats, dim=1)
     
        contents = self.fusion_net(feats_all.permute(0,2,3,1)).permute(0,3,1,2)
        
            
        for j in range(4):
            feats_pred[j] = contents*codes[j]
            preds[j] = self.imnet(feats_pred[j].permute(0,2,3,1)).permute(0,3,1,2)
        
        return preds, feats

                
        
       
