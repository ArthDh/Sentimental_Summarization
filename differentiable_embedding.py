import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F

class Embedding_(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding_, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.embedding.weight)
