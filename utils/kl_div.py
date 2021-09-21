# Apply the weighting lambdas in the main function this is just a loss without lambda weights
def kl_div_loss(p_pred, p_target):
    
    softmax = nn.Softmax(dim=-1)
    
    logsoftmax = nn.LogSoftmax(dim=1)
    
    return torch.mean(torch.sum(- softmax(p_target) * logsoftmax(p_pred), -1))
