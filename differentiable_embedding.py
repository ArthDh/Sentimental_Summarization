import torch
import torch.autograd
from torch.autograd import Variable

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.weights = torch.randn((vocab_size, embedding_dim), requires_grad=True)

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.weights[mask]
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.weights)

vocab_size = 3
sent_len = 2
embedding_dim = 5
batch_size = 10

# get logits
x = Variable(torch.randint(0,10,(batch_size, sent_len, vocab_size)).float(), requires_grad = True)
y = Variable(torch.randint(0,10,(batch_size, sent_len, vocab_size)).float(), requires_grad = True)
z = y - 10
t = x + 6
logits = t/z

# Masks
idx =  torch.argmax(logits, dim=-1, keepdims=  True)
mask = torch.zeros_like(logits).scatter_(-1, idx, 1.).float().detach() + logits - logits.detach()

# Find embedding
emb_layer = Embedding(vocab_size, embedding_dim)

embedding = emb_layer(mask)

# Define some loss function to test the autograd
loss = torch.sum(embedding)

# test autograd
print(torch.autograd.grad(loss, mask, retain_graph = True))
