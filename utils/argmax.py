import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# get logits
x = Variable(torch.randn(10, 3), requires_grad = True)
y = Variable(torch.randn(10, 3), requires_grad = True)
z = y*x + 6
t = x- y + z
logits = t/z * torch.sum(x, -1).view(10,1)


prob_logits,_ = torch.max(logits,-1)

# The gumbel inspired implementation
ids = torch.argmax(logits, -1).float() + prob_logits - prob_logits.detach()

l = torch.ones(ids.shape)

loss = torch.dot(ids,l)

print(torch.autograd.grad(loss, logits, retain_graph=True))