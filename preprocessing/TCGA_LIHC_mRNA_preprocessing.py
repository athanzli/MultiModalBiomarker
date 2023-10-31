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

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_random_seed(42)

#%% ################################## data

# mrna
mrna_data = pd.read_table('./data/TCGA-LIHC/HiSeqV2', index_col=0)
mrna_data = mrna_data.T
# survival
surv_data = pd.read_table('./data/TCGA-LIHC/survival_LIHC_survival.txt', index_col=0)
# clinical 
clic_data = pd.read_table('./data/TCGA-LIHC/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix', index_col=0)

r""" interested columns:

sample_type
Primary Tumor          377
Solid Tissue Normal     59
Recurrent Tumor          2

pathologic_stage
Stage I          198
Stage II         100
Stage IIIA        74
Stage IIIB        10
Stage IIIC        10
Stage III          6
Stage IV           3
[Discrepancy]      2
Stage IVB          2
Stage IVA          1

adjacent_hepatic_tissue_inflammation_extent_type
Mild      116
Severe     22
"""

#%% sample patients
clic = clic_data[~(clic_data['sample_type']=='Recurrent Tumor')].copy()
patients = np.intersect1d(np.intersect1d(mrna_data.index, surv_data.index), clic.index)
pos_sel = clic['sample_type'][clic['sample_type']=='Primary Tumor'].sample(50, random_state=0).index
neg_sel = clic['sample_type'][clic['sample_type']=='Solid Tissue Normal'].index
patients = np.intersect1d(patients, np.concatenate([pos_sel, neg_sel]))
clic = clic.loc[patients, :]
mrna_data = mrna_data.loc[patients, :]
surv_data = surv_data.loc[patients, :]

y = clic['sample_type']
y = y.map({'Primary Tumor': 1, 'Solid Tissue Normal': 0}).values

#%% preprocessing mrna data
assert ~mrna_data.isna().any().any() # check missing values
mrna_data = mrna_data.loc[~mrna_data.index.duplicated()] # remove duplicated rows
mrna_data = mrna_data.loc[:,~mrna_data.columns.duplicated()] # remove duplicated features
mrna_data = mrna_data.iloc[:, ~((mrna_data == 0).sum(axis=0)==mrna_data.shape[0]).values] # remove features with 0 across all samples

#%% DEGs using R
# mrna_data_sel = mrna_data.loc[:, (mrna_data.mean(axis=0)>1).values] # NOTE?


#%% scaling
X = StandardScaler().fit_transform(mrna_data_sel)

#%% temp, look at data stats
pvals = {}
for gene in mrna_data.columns:
    _, p = scipy.stats.mannwhitneyu(
        mrna_data[gene][y==1], 
        mrna_data[gene][y==0])
    pvals[gene] = p
#%%
pvals = pd.Series(pvals)
pvals = pvals.sort_values()
pvals = pvals[pvals<0.05]
pvals=pvals.to_frame()
pvals = pvals.rename(columns={0:'p'})
pvals['rank'] = np.arange(pvals.shape[0])

pvals.loc['AFP']

#%% train test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(sss.split(X, y))
X_trn, X_tst = X[train_idx], X[test_idx]
y_trn, y_tst = y[train_idx], y[test_idx]


#%% ################################ model
model = MMBL(
    seq_len = X.shape[1],
    emb_dim = X.shape[1] // 2,
    mlp_channels = [X.shape[1], X.shape[1], X.shape[1] // 2, 1],
    n_heads = 1
)

#%% ############################### train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
n_epochs = 20

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
    print(f'tst accuracy {acc.item():.3f}')
    tst_acc.append(acc.item())

# print('Test MSE', torch.mean((logits.squeeze().detach() - y_tst)**2).item())

# plot
_, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(trn_losses, label='train')
axes[0].plot(tst_losses, label='test')
axes[0].set_ylim(0, 1)
axes[0].set_title('loss')
axes[0].legend()
axes[1].plot(trn_acc, label='train')
axes[1].plot(tst_acc, label='test')
axes[1].set_ylim(0, 1)
axes[1].set_title('accuracy')
axes[1].legend() 
plt.tight_layout() 
plt.show()

#%% #################################### analysis
## See biomarker genes
def att_cat(A, cat_mask):
    A_avg = A[cat_mask].mean(0)
    feature_attn = A_avg.mean(0)
    return A_avg, feature_attn
def plot_ft_imp(ft_imp, n_fts, title=None):
    _, ax = plt.subplots(figsize=(15, 5))
    argsort = np.argsort(ft_imp)[::-1]
    sns.barplot(
        x=np.arange(n_fts),
        y=ft_imp[argsort], 
        ax=ax
    )
    ax.set_xticklabels(argsort)
    plt.title(title)
    plt.show()
    return argsort

#%% get effective attention
model.eval()
A_eff = model.module1.effective_attention(X_trn).detach().numpy()

#%% plot effective attention
A_avg, ft_imp = att_cat(A_eff, y_trn==1)
ranked_ft_idx = plot_ft_imp(ft_imp, X.shape[1])
print(ft_names[ranked_ft_idx])
A_avg, ft_imp = att_cat(A_eff, y_trn==0)
ranked_ft_idx = plot_ft_imp(ft_imp, X.shape[1])
print(ft_names[ranked_ft_idx])

#%% analyse attention scores
sns.heatmap(A_avg, cmap='Reds', ); plt.show()