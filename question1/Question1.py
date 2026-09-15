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


def sparse_attention(Q,K,V,window):

    scores = Q @ K.transpose(-2,-1)
    scores = scores / sqrt(Q.size(-1))

    scores = scores.masked_fill(~sliding_window_mask(Q.size(-2), window), float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ V


