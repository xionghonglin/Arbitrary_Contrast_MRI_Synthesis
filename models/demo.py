import torch
import torch.nn as nn

import torch.nn.functional as F
#import numpy as np
from models import register
from simple_tokenizer import SimpleTokenizer

if __name__ == '__main__':
    def tokenize(tokenizer, text, context_length):
       
        tokens = tokenizer.encode(text) 
        result = torch.zeros(context_length)
        result[:len(tokens)] = torch.tensor(tokens)
  
        return result
    text_lib= ['T1','T2w','FLAIR','Gd']
    tokenizer = SimpleTokenizer()
    for text in text_lib:
        print(tokenize(tokenizer, text, 4))
