# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:56:59 2018

@author: DELL
"""

import torch

def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7)
features=torch.randn((1,5))
weights=torch.randn_like(features)
bias=torch.randn((1,1))
y=activation(torch.mm(features,weights.view(5,1))+bias)
print(y)

 
