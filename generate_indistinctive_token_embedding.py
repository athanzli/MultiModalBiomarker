import torch
SAVE_PATH = './saved'

L = 5995 + 1
# L = 55 + 1
E = 192

min_val = -4.7
max_val = 4.7

tensor = torch.ones(L, E)

# tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
# tensor = tensor * (max_val - min_val) + min_val

# tensor[0] = torch.zeros(E)

tensor=tensor.to('cuda:0')

torch.save(tensor, SAVE_PATH + f'/indistinctive_pretrained_token_emb_weight_mRNA+miRNA.pt')