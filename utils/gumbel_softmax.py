import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# get logits
x = Variable(torch.randn(10, 3), requires_grad = True)
y = Variable(torch.randn(10, 3), requires_grad = True)
z = y - 10
t = x + 6
logits = t/z

# Sample hard categorical using "Straight-through" trick:
out = F.gumbel_softmax(logits, tau=0.1, hard=True)

# loss
l = torch.randn(10,3)
loss = torch.sum(out*l)

grad_x = torch.autograd.grad(loss, x, retain_graph=True)
grad_y = torch.autograd.grad(loss, y, retain_graph=True)

