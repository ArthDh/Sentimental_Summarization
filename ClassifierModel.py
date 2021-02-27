import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn.functional as F

class Embedding_(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding_, self).__init__()
        
        self.embedding = torch.nn.Embedding.from_pretrained(self.weights)

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.embedding.weight)

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
        self.embedding = Embedding_(vocab_size, embedding_dim).requires_grad_()
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        
        fc_hidden = 32
        self.fc1 = torch.nn.Linear(hidden_size*seq_length, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, out_dim)
        

    def forward(self, x):
        embedding_out = self.embedding(x)
        out, _ = self.lstm(embedding_out)
        fc1 = F.relu(self.fc1(out.reshape(x.shape[0], -1)))
        return self.fc2(fc1)

    def loss(self, x, target):

        logits = self(x)

        return loss_fn(logits, target)
