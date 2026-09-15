import torch
from math import sqrt

def sliding_window_mask(T, window):

    i=torch.arange(T).unsqueeze(1)
    j=torch.arange(T).unsqueeze(0)

    mask = (j<=i) & (j>=i-window+1)

    return mask

def block_space_mask(T,b_size):

    token=torch.arange(T)

    block_id=token//b_size

    i = torch.unsqueeze(block_id,1)
    j= torch.unsqueeze(block_id,0)

    mask = j <= i

    return mask

def block_space_mask_Bigbird(T, b_size, window,global_token,num_rand):

    token=torch.arange(T)
    block_id=token//b_size

    num_blocks= (T+b_size-1)//b_size

    local_mask=sliding_window_mask(T,window)

    i=torch.arange(T).unsqueeze(1)
    j=torch.arange(T).unsqueeze(0)

    global_mask = (i==global_token) | (j==global_token)

    i = torch.unsqueeze(block_id,1)
    j= torch.unsqueeze(block_id,0)

    random_block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)

    for block in range(num_blocks):

        possible_blocks= torch.arange(block+1)
        shuffled = possible_blocks[torch.randperm(len(possible_blocks))]
        chosen = shuffled[:num_rand]

        random_block_mask[block, chosen] = True

    random_mask = torch.zeros(T,T, dtype=torch.bool)

    for query_token in range(T):
        for key_token in range(T):

            query_block = query_token//b_size
            key_block = key_token//b_size

            if random_block_mask[query_block,key_block]:
                random_mask[query_token,key_token] = True
        

    mask = local_mask | global_mask | random_mask
    
    return mask

def dense_attention(Q,K,V,mask):
    scores = Q @ K.transpose(-2,-1)
    scores = scores / sqrt(Q.size(-1))
    
    fully_masked_rows = []

    for row in mask:
        zero_count=0

        for i in row:
            if i == False:
                zero_count+=1
        if zero_count == Q.size(-2):
            fully_masked_rows.append(True)
        else:
            fully_masked_rows.append(False)

        #or 
        # row = torch.tensor([False, False, False])
        # torch.all(row == False)

    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    for i in range(Q.size(-2)):
        if fully_masked_rows[i]:
            weights[:, :, i, :] = 0

    return weights @ V


def sparse_attention(Q,K,V,window):

    scores = Q @ K.transpose(-2,-1)
    scores = scores / sqrt(Q.size(-1))

    mask = sliding_window_mask(Q.size(-2), window)

    fully_masked_rows = []


    for row in mask:
        zero_count=0

        for i in row:
            if i == False:
                zero_count+=1
        if zero_count == Q.size(-2):
            fully_masked_rows.append(True)
        else:
            fully_masked_rows.append(False)

        #or 
        # row = torch.tensor([False, False, False])
        # torch.all(row == False)

    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    for i in range(Q.size(-2)):
        if fully_masked_rows[i]:
            weights[:, :, i, :] = 0

    return weights @ V

def main():


    return 0

