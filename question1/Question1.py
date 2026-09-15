import torch
from math import sqrt
from memory_profiler import memory_usage
import time
import matplotlib.pyplot as plt

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
    
    causal_mask = torch.tril(
        torch.ones(T, T, dtype=torch.bool)
    )

    mask = mask & causal_mask

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
            weights[ i, :] = 0

    return weights @ V

def sliding_sparse_attention(Q, K, V, window):

    T = Q.size(0)
    D = Q.size(1)

    output = torch.zeros(T, D)

    for i in range(T):

        start = max(0, i - window + 1)
        end = i + 1

        K_allowed = K[start:end]
        V_allowed = V[start:end]

        scores = Q[i] @ K_allowed.transpose(-2, -1)

        scores = scores / sqrt(D)

        weights = torch.softmax(scores, dim=-1)

        output[i] = weights @ V_allowed

    return output

def block_sparse_attention(Q, K, V, b_size=4, window=3, global_token=0, num_rand=1):

    T = Q.size(0)
    D = Q.size(1)

    mask = block_space_mask_Bigbird(T,b_size,window,global_token,num_rand)

    output = torch.zeros(T, D)

    for i in range(T):

        allowed = mask[i]

        K_allowed = K[allowed]
        V_allowed = V[allowed]

        scores = Q[i] @ K_allowed.transpose(-2, -1)

        scores = scores / sqrt(D)

        weights = torch.softmax(scores, dim=-1)

        output[i] = weights @ V_allowed

    return output

def benchmark_time(Q,K,V,mask):

    time_taken=[]

    start = time.perf_counter()

    dense_attention(Q,K,V,mask)

    end = time.perf_counter()

    time_taken.append(end - start)

    start = time.perf_counter()
    
    block_sparse_attention(Q,K,V,2)

    end = time.perf_counter()

    time_taken.append(end - start)

    start = time.perf_counter()
        
    sliding_sparse_attention(Q,K,V,3)

    end = time.perf_counter()

    time_taken.append(end - start)

    return time_taken

def benchmark_memory(Q,K,V,mask):

    memory=[]

    memory_used = memory_usage(
        (dense_attention, (Q, K, V, mask)),
        max_usage=True
    )
    memory.append(memory_used)


    memory_used = memory_usage(
        (block_sparse_attention, (Q, K, V, 2)),
        max_usage=True
    )
    memory.append(memory_used)

    memory_used = memory_usage(
        (sliding_sparse_attention, (Q, K, V, 3)),
        max_usage=True
    )
    memory.append(memory_used)

    return memory

#part of question 1.6

# def data_prep():

#     with open("input.txt", "r", encoding="utf-8") as f:
#     text = f.read()

#     print("Length of text:", len(text))

#     chars=sorted(list(set(text)))

#     print(len(chars))
#     print(chars)


def main():
    T_values = [512, 1024, 2048, 4096]
    D = 4

    for T in T_values:
        Q = torch.randn(T, D)
        K = torch.randn(T, D)
        V = torch.randn(T, D)

        mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool)
        )

        time_taken = benchmark_time(Q, K, V, mask)

        memory_used = benchmark_memory(Q,K,V,mask)


        print("time taken for dense:", time_taken[0], "time taken for block_sparse:", time_taken[1], "time taken for sliding_sparse:", time_taken[2])
        print("memory used for dense:", memory_used[0],"memory used for block_sparse:", memory_used[1], "memory used for sliding_sparse", memory_used[2])

    


# if __name__ == "__main__":
#     main()

T_values = [512, 1024, 2048, 4096]

time_taken_dense = [
    1.0442,
    4.1212,
    14.5444,
    96.7555
]

time_taken_block_sparse = [
    0.8814,
    3.5340,
    12.6340,
    70.3206
]

time_taken_sliding_sparse = [
    0.0212,
    0.0292,
    0.0502,
    0.1006
]

plt.figure(figsize=(8, 5))

plt.plot(
    T_values,
    time_taken_dense,
    marker="o",
    label="Dense Attention"
)

plt.plot(
    T_values,
    time_taken_block_sparse,
    marker="o",
    label="BigBird Sparse Attention"
)

plt.plot(
    T_values,
    time_taken_sliding_sparse,
    marker="o",
    label="Sliding Window Attention"
)

plt.xlabel("Sequence Length (T)")
plt.ylabel("Time (seconds)")
plt.title("Attention Runtime vs Sequence Length")

plt.legend()
plt.grid(True)

plt.show()
