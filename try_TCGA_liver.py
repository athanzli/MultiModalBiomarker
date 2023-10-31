#%%
%load_ext autoreload
%autoreload 2
import torch
import pandas as pd
import random, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import scipy
import sklearn
from sklearn import linear_model
import shap
#
import utils
from model import MMBL, MLP

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_random_seed(42)

#%% ################################# data
#%% ################################# data: mRNA
# mrna
mrna = pd.read_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts.csv', index_col=0)
tumor_type_codes = [int(barcode.split('-')[3][:-1]) for barcode in mrna.columns.values] # https://docs.gdc.cancer.gov/Encyclopedia/pages/images/TCGA-TCGAbarcode-080518-1750-4378.pdf
y = pd.Series(tumor_type_codes).map({1: 'tumor', 2: 'tumor', 11:'normal'}).values # 1 for solid tumor, 0 for normal
print(np.unique(y, return_counts=True))
pd.DataFrame(index=mrna.columns.values, columns=['sample_type'], data=y).to_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_sample_type.csv')
X = mrna.T
# # remove features that have 0 values across more than 20% samples
# X = X.iloc[:, ~((X == 0).sum(axis=0)>X.shape[0]*0.2).values]
# log normalization
X_raw = X
X = np.log(X+1)

# split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(sss.split(X, y))
X_trn, X_tst = X.iloc[train_idx], X.iloc[test_idx]
y_trn, y_tst = y[train_idx], y[test_idx]
print(np.unique(y_trn, return_counts=True))
X_trn.T.to_csv("./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_trn.csv")
pd.DataFrame(index=X_trn.index.values, columns=['sample_type'], data=y_trn).to_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_sample_type_trn.csv')

# deg set
deg = pd.read_csv('./data/TCGA-LIHC_from_GDC_raw/DEG_trn.csv', index_col=0)
deg_set = deg[(deg['padj']<0.05) & (deg['log2FoldChange'].abs()>2)].index
X_trn = X_trn[deg_set]
X_tst = X_tst[deg_set]
assert all((X_trn.columns==X_tst.columns) & (X_trn.columns==deg_set))
y_trn = pd.Series(y_trn).map({'tumor': 1, 'normal': 0}).values
y_tst = pd.Series(y_tst).map({'tumor': 1, 'normal': 0}).values
ft_names = X_trn.columns.values

# scaling
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# #%% ################################# data: specimen
# biopsy = pd.read_table("./data/TCGA-LIHC_from_GDC_raw/biospecimen.car.2023-10-30/sample.tsv", index_col=0)
# biopsy['sample_type'].value_counts()
# tumor_type_info = biopsy.groupby(['case_submitter_id', 'sample_type']).size().unstack()
# (~tumor_type_info['Primary Tumor'].isna()).sum()
# (~tumor_type_info['Blood Derived Normal'].isna()).sum()
# (~tumor_type_info['Solid Tissue Normal'].isna()).sum()

#%% ################################# data: clinical
r"""
- the reason why there are overlapping rows (patients) is that some patients have multiple rows recording differnt level of information (e.g., treamtent)
"""
clinic = pd.read_table('./data/TCGA-LIHC_from_GDC_raw/clinical.cart.2023-10-30/clinical.tsv', index_col=0)
clinic = clinic[['case_submitter_id', 'project_id', 'age_at_index', 'days_to_death', 'days_to_last_follow_up', 'vital_status', 'treatment_or_therapy', 'treatment_type']]
# mrna_samples = ['-'.join(sample_id.split('-')[:3]) for sample_id in mrna.index]

# The data from UCSC
# clinic = pd.read_table("C:/Users/athan/OneDrive/Desktop/MMBiomarker/MMBiomarker/data/TCGA-LIHC_from_UCSC/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix")
# tumor_type_info = clinic.groupby(['_PATIENT', 'sample_type']).size().unstack()
# (~tumor_type_info['Primary Tumor'].isna()).sum()
# (~tumor_type_info['Solid Tissue Normal'].isna()).sum()
# (~tumor_type_info['Recurrent Tumor'].isna()).sum()
# mrna = pd.read_talbe('C:\Users\athan\OneDrive\Desktop\MMBiomarker\MMBiomarker\data\TCGA-LIHC_from_UCSC\HiSeqV2')

#%% ################################# data: survival
# TODO use days_to_death, days_to_last_follow_up, vital status
# OS:
#   for dead patients, OS = days_to_death
#   for alive patients, OS = days_to_last_follow_up
#   if not reported, then OS is NA
# Censor: 
#   if alive, then Censor is labeled as 0
#   if dead, then Censor is labeled as 1
#   if not reported, then Censor is NA

#%% ################################ model
model = MMBL(
    seq_len = X_trn.shape[1],
    emb_dim = X_trn.shape[1] // 2,
    mlp_channels = [X_trn.shape[1], X_trn.shape[1], X_trn.shape[1] // 2, 1],
    n_heads = 1,
    n_encoders = 2,
)

#%% ############################### train
# TODO  change accordingly
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.MSELoss()

X_trn = utils.convert2tensor(X_trn)
y_trn = utils.convert2tensor(y_trn)
X_tst = utils.convert2tensor(X_tst)
y_tst = utils.convert2tensor(y_tst)

trn_losses = []
tst_losses = []
trn_acc = []
tst_acc = []

trn_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_trn, y_trn),
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

#%% 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

n_epochs = 5
for epoch in range(n_epochs):
    model.train()

    # TODO batch training
    # epoch_loss = 0
    # for batch in trn_loader:
    #     X_trn, y_trn = batch
    #     # TODO change accordingly
    #     logits = model(X_trn)
    #     loss = loss_fn(logits.squeeze(), y_trn)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     epoch_loss += loss.item()
    # epoch_loss /= len(trn_loader)
    # trn_losses.append(epoch_loss)

    logits = model(X_trn)
    loss = loss_fn(logits.squeeze(), y_trn)
    trn_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO change accordingly
    acc = (logits.sigmoid().round() == y_trn.unsqueeze(1)).float().mean()
    print(f'epoch {epoch}, loss {loss.item()}, trn accuracy {acc.item()}')
    # print(f'epoch {epoch}, loss {loss.item()}')
    trn_acc.append(acc.item())

    # eval
    model.eval()
    # TODO change accordingly
    # logits = model(X_tst).detach()
    logits = model(X_tst).detach().squeeze()
    loss = loss_fn(logits.squeeze(), y_tst)
    tst_losses.append(loss.item())
    acc = (logits.sigmoid().round() == y_tst).float().mean()
    print(f'val accuracy {acc.item():.3f}')
    tst_acc.append(acc.item())

# print('Test val', torch.mean((logits.squeeze().detach() - y_tst)**2).item())

# plot
_, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(trn_losses, label='train')
axes[0].plot(tst_losses, label='val')
axes[0].set_ylim(0, 1)
axes[0].set_title('loss')
axes[0].legend()
axes[1].plot(trn_acc, label='train')
axes[1].plot(tst_acc, label='val')
axes[1].set_ylim(0, 1)
axes[1].set_title('accuracy')
axes[1].legend() 
plt.tight_layout() 
plt.show()

#%% #################################### analysis
## See biomarker genes
def att_cat(A, cat_mask):
    A_avg = np.median(A[cat_mask], axis=0)
    feature_attn = np.mean(A_avg, axis=0)
    return A_avg, feature_attn
def plot_ft_imp(ft_names, ft_imp, title=None):
    sorted_idx = sorted(range(len(ft_imp)), key=lambda k: ft_imp[k], reverse=True)
    ft_names = [ft_names[i] for i in sorted_idx]
    ft_imp = [ft_imp[i] for i in sorted_idx]

    plt.figure(figsize=(10, 3))
    bars = plt.bar(range(len(ft_imp)), ft_imp, align='center')
    plt.xticks(range(len(ft_imp)), ft_names, rotation='vertical')

    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.title(title)
    plt.show()

#%% get effective attention
model.eval()
A_eff = model.module1.effective_attention(X_trn).detach().numpy()

#%% plot effective attention
A_avg, ft_imp = att_cat(A_eff, y_trn==1)
ranked_ft_idx = plot_ft_imp(ft_names[:50], ft_imp[:50]) # show only top 50
A_avg, ft_imp = att_cat(A_eff, y_trn==0)
ranked_ft_idx = plot_ft_imp(ft_names[:50], ft_imp[:50])

#%% analyse attention scores
# sns.heatmap(A_avg[:50,:50], cmap='Reds', ); plt.show()


#%% #################################### backup code
idx=[]
for i in range(clinic.shape[1]):
    if (clinic.iloc[:, i].values[3]=="'--") & (clinic.iloc[:, i].values[35]=="'--") & (clinic.iloc[:, i].values[66]=="'--") & (clinic.iloc[:, i].values[111]=="'--"):
        idx.append(i)
clinic = clinic.drop(columns=clinic.columns[idx])
