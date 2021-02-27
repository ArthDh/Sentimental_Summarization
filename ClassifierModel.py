import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.weights = torch.randn((vocab_size, embedding_dim), requires_grad=True).cuda()

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.weights[mask]
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.weights)

class ClassifierModel(torch.nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim,
                 out_dim,
                 n_layers,
                 hidden_size,
                 dropout=0.5,
                 batch_first=True,
                 seq_length = 150
                ):
        super(ClassifierModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, embedding_dim).requires_grad_()
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        
        self.fc1 = torch.nn.Linear(hidden_size*seq_length, 32)
        self.fc2 = torch.nn.Linear(32, out_dim)
        

    def forward(self, x):
        embedding_out = self.embedding(x)
        out, _ = self.lstm(embedding_out)
        fc1 = F.relu(self.fc1(out.reshape(x.shape[0], -1)))
        return self.fc2(fc1)
