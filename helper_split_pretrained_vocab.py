# just some temp helper code

import torch
import pandas as pd

DATA_PATH = '/home/che82/athan/MMBiomarker/saved' # DATA_PATH = 'C:/Users/athan/OneDrive/Desktop/MMBiomarker/saved/saved'

N_TOKEN_MRNA = 4222
N_TOKEN_MIRNA = 1773

pretrained_token_emb_weight = torch.load(
    # DATA_PATH + '/pretrained_token_emb_weight_mRNA+miRNA.pt', 
    DATA_PATH + '/random_pretrained_token_emb_weight_mRNA+miRNA.pt', 
    map_location=torch.device('cpu'))

pretrained_token_emb_weight_mrna = pretrained_token_emb_weight[1:(N_TOKEN_MRNA+1), :]
pretrained_token_emb_weight_mirna = pretrained_token_emb_weight[(N_TOKEN_MRNA+1):(1+N_TOKEN_MRNA+N_TOKEN_MIRNA), :]

pretrained_token_emb_weight_mrna = torch.vstack([
    torch.zeros((1, pretrained_token_emb_weight_mrna.shape[1])),
    pretrained_token_emb_weight_mrna
])
pretrained_token_emb_weight_mirna = torch.vstack([
    torch.zeros((1, pretrained_token_emb_weight_mirna.shape[1])),
    pretrained_token_emb_weight_mirna
])

# torch.save(pretrained_token_emb_weight_mrna, DATA_PATH + '/pretrained_token_emb_weight_mRNA.pt')
# torch.save(pretrained_token_emb_weight_mirna, DATA_PATH + '/pretrained_token_emb_weight_miRNA.pt')
torch.save(pretrained_token_emb_weight_mrna, DATA_PATH + '/random_pretrained_token_emb_weight_mRNA.pt')
torch.save(pretrained_token_emb_weight_mirna, DATA_PATH + '/random_pretrained_token_emb_weight_miRNA.pt')

