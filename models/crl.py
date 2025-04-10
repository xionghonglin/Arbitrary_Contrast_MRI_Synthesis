import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
from models.linear import Linear
import models
from models import register

@register('crl')
class crl(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        f_dim = self.encoder.out_dim
        self.codemapping_mlp = Linear(in_dim=3, out_dim=f_dim, hidden_list = [f_dim, f_dim, f_dim, f_dim])     
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat
    
    def get_code(self, inp):
        code = self.codemapping_mlp(inp)
        return code
    
    def modal_trans(self, feat_src, code_src, code_tgt):
  
        latent_content = feat_src/code_src.unsqueeze(-1).unsqueeze(-1)
        feat_tgt = latent_content*code_tgt.unsqueeze(-1).unsqueeze(-1)
        
        pred_tgt = self.imnet(feat_tgt.permute(0,2,3,1)).permute(0,3,1,2)    
                
        return pred_tgt, latent_content

    def forward(self, img_src, cond_src, cond_tgt):
        
        feat_src = self.gen_feat(img_src)
        code_src = self.get_code(cond_src)
        code_tgt = self.get_code(cond_tgt)
        
        pred, content= self.modal_trans(feat_src, code_src, code_tgt)

        return pred,content
