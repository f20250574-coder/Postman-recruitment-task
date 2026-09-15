import torch
from math import sqrt


def attention(Q,K,V,mask):
    scores = Q @ K.transpose(-2,-1)
    scores = scores / sqrt(Q.size(-1))

    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ V

def sliding_window_mask(T, window):

    i=torch.arange(T).unsqueeze(1)
    j=torch.arange(T).unsqueeze(0)

    mask = (j<=i) & (j<=i-window+1)

    return mask

def block_space_mask(T,b_size):

    token=torch.arange(T)

    block_id=i//b_size

    i = torch.unsqueeze(block_id,1)
    j= torch.unsqueeze(block_id,0)

    mask = j <= i

    return mask

def block_space_mask_Bigbird(T, b_size, window,global_token,num_rand):

    token=torch.arange(T)

    local_mask=sliding_window_mask(T,window)

    i=torch.arange(T).unsqueeze(1)
    j=torch.arange(T).unsqueeze(0)

    block_id=i//b_size

    global_mask = (i==global_token) | (j==global_token)

    i = torch.unsqueeze(block_id,1)
    j= torch.unsqueeze(block_id,0)

    random_mask = torch.zeros(T, T, dtype=torch.bool)








def sparse_attention(Q,K,V,window):

    scores = Q @ K.transpose(-2,-1)
    scores = scores / sqrt(Q.size(-1))

    scores = scores.masked_fill(~sliding_window_mask(Q.size(-2), window), float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ V



