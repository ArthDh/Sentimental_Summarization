def soft_arg_max(A, mask = 1, beta=1e5, dim=2, epsilon=1e-12):
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp((A - A_max)*beta)
    A_exp = A_exp * mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float()
    return torch.matmul(A_softmax, indices)