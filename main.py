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

#%% ################################## data: mRNA liver TCGA
# mrna
mrna_data = pd.read_table('./data/TCGA-LIHC/HiSeqV2', index_col=0)
mrna_data = mrna_data.T
# survival
surv_data = pd.read_table('./data/TCGA-LIHC/survival_LIHC_survival.txt', index_col=0)
# clinical 
clic_data = pd.read_table('./data/TCGA-LIHC/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix', index_col=0)
r"""3 interested columns:

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

mrna_data.var(axis=0).hist(bins=100); plt.show()
hvgs = mrna_data.var(axis=0).sort_values(ascending=False)[:2000] # TODO
mrna_data_sel = mrna_data.loc[:, hvgs.index]
# X = StandardScaler().fit_transform(mrna_data) # TODO

# scaling
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

#%% #################################### mdoel
model = MMBL(
    n_genes = X_trn.shape[1],
    emb_dim = 64,
    mlp_channels = [X_trn.shape[1], 128, 1],
    n_heads = 1,
)

# model = MLP([X_trn.shape[1], 128, 128, 128, 1])

# model = MMBL(
#     n_genes = X_trn.shape[1],
#     emb_dim = 8,
#     mlp_channels = [X_trn.shape[1], 32, 32, 1],
#     n_heads = 1,
# )

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
n_epochs = 20

#%% training
# TODO  change accordingly
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.MSELoss()

X_trn = utils.convert2tensor(X_trn)
y_trn = utils.convert2tensor(y_trn)
X_tst = utils.convert2tensor(X_tst)
y_tst = utils.convert2tensor(y_tst)

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # TODO change accordingly
    logits = model(X_trn)
    loss = loss_fn(logits.squeeze(), y_trn)
    # logits = model(X_trn)
    # loss = loss_fn(logits.squeeze(), y_trn)

    loss.backward()
    optimizer.step()

    # TODO change accordingly
    accuracy = (logits.sigmoid().round() == y_trn.unsqueeze(1)).float().mean()
    print(f'epoch {epoch}, loss {loss.item()}, trn accuracy {accuracy.item()}')
    # print(f'epoch {epoch}, loss {loss.item()}')

# test accuracy
model.eval()
# TODO change accordingly
# logits = model(X_tst).detach()
logits = model(X_tst).detach().squeeze()
accuracy = (logits.sigmoid().round() == y_tst).float().mean()
print(f'tst accuracy {accuracy.item():.3f}')
# print('Test MSE', torch.mean((logits.squeeze().detach() - y_tst)**2).item())

#%% [markdown]
## See biomarker genes
model.eval()
_, attn = model.module1(X_trn, output_attn_scores=True)
attn = attn.detach().numpy()

# mask = (y_trn==1).numpy()
# attn_pos = attn[mask].mean(0)
# attn_ctl = attn[~mask].mean(0)
attn = attn.mean(0)

# gene_attn_pos = attn_pos.mean(0)
# gene_attn_ctl = attn_ctl.mean(0)
gene_attn = attn.mean(0)

# df = mrna_data_sel
# attn = pd.DataFrame(attn, index=df.columns, columns=df.columns)
# attn_pos = pd.DataFrame(attn_pos, index=df.columns, columns=df.columns)
# attn_ctl = pd.DataFrame(attn_ctl, index=df.columns, columns=df.columns)
#%% analyse attention scores
# sns.heatmap(attn, cmap='Reds', ); plt.show()
# sns.heatmap(attn_pos, cmap='Reds', ); plt.show()
# sns.heatmap(attn_ctl, cmap='Reds', ); plt.show()

# sns.barplot(
#     x=df.columns, 
#     y=gene_attn_pos
# )
# sns.barplot(
#     x=df.columns, 
#     y=gene_attn_ctl
# )
# sns.barplot(
#     x=df.columns, 
#     y=gene_attn
# )

# rank genes by attention scores
df.columns[np.argsort(gene_attn)[::-1]]
df.columns[np.argsort(gene_attn_pos)[::-1]]
df.columns[np.argsort(gene_attn_ctl)[::-1]]

np.where(df.columns[np.argsort(gene_attn)[::-1]] == 'AFP')[0]



























#%% ############################## toy datasets
#%%
from sklearn import linear_model
import shap
X = torch.randn(200, 50)
y = torch.randint(0, 2, (200,))
X[:, 3][y==1] = 1
X[:, 3][y==0] = -1
X[:, 5][y==1] = -1
X[:, 5][y==0] = 1
X[:, 8][y==1] = 1
X[:, 8][y==0] = -1

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(sss.split(X, y))
X_trn, X_tst = X[train_idx], X[test_idx]
y_trn, y_tst = y[train_idx], y[test_idx]

#%%
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(X_trn, y_trn)
def print_accuracy(f):
    accuracy = np.mean(f(X_tst) == y_tst)
    print("Accuracy on test set = {:.2f}%".format(accuracy * 100))

print_accuracy(logistic_regr.predict)
ex = shap.KernelExplainer(logistic_regr.predict_proba, X_trn.numpy())
shap_values = ex.shap_values(X_tst.numpy())
shap.summary_plot(shap_values[1], X_tst.numpy(), plot_type="bar", title="Shapley Values for Positive Class")

#%% out model
model = MMBL(
    n_genes = X.shape[1],
    emb_dim = 16,
    mlp_channels = [X.shape[1], X.shape[1], 1],
    n_heads = 1,
)
#%%

#%%  ####################### diabetes
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
import time
from sklearn.model_selection import train_test_split
import shap

X, y = shap.datasets.diabetes()
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=0)

#%% shapley
# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_trn_summary = shap.kmeans(X_trn, 10)

def print_accuracy(f):
    print(
        "Root mean squared test error = {}".format(
            np.sqrt(np.mean((f(X_tst) - y_tst) ** 2))
        )
    )
    time.sleep(0.5)  # to let the print get out before any progress bars

shap.initjs()

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_trn, y_trn)

print_accuracy(lin_regr.predict)

ex = shap.KernelExplainer(lin_regr.predict, X_trn_summary)
shap_values = ex.shap_values(X_tst)
shap.summary_plot(shap_values, X_tst)

#%% model
model = MMBL(
    n_genes = X.shape[1],
    emb_dim = 8,
    mlp_channels = [X.shape[1], 16, 1],
    n_heads = 1,
)
#%%
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
n_epochs = 20

#%% training
# TODO  change accordingly
loss_fn = torch.nn.MSELoss()

X_trn = utils.convert2tensor(X_trn)
y_trn = utils.convert2tensor(y_trn)
X_tst = utils.convert2tensor(X_tst)
y_tst = utils.convert2tensor(y_tst)

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # TODO change accordingly
    logits = model(X_trn)
    loss = loss_fn(logits.squeeze(), y_trn)

    loss.backward()
    optimizer.step()

    # TODO change accordingly
    print(f'epoch {epoch}, loss {loss.item()}')

# test accuracy
model.eval()
# TODO change accordingly
logits = model(X_tst).detach().squeeze()
print('Test MSE', torch.mean((logits.squeeze().detach() - y_tst)**2).item())

#%%















# %% [markdown]
## titanic
df = pd.read_csv('./data/titanic.csv', index_col=0)
# extracting the title from the name:
Title = []
for name in  df.Name:
    Title.append(name.split(",")[1].split(".")[0])
df["Title"] = Title
#grouping people with pclass and title
df.groupby(["Pclass", 'Title'])['Age'].agg(['mean']).round(0)
# adding the mean of the age of each group to the missing values
df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(lambda x: x.fillna(x.mean()))

df = df.drop(columns=["Cabin"])
df = df.drop(columns = ["Name"])
df = df.drop(columns = ["Ticket"])
df = df.drop(columns = ["Title"])
df.Sex = pd.Categorical(df.Sex)
df.Embarked = pd.Categorical(df.Embarked)
df["Sex"] = df.Sex.cat.codes
df["Embarked"] = df.Embarked.cat.codes
target = df.Survived.values # NOTE
df = df.drop(columns =["Survived"])

from sklearn.model_selection import train_test_split
X_trn, X_tst, y_trn, y_tst = train_test_split(df, target, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_trn, y_trn)
LR.score(X_tst, y_tst)

import shap
explainer = shap.LinearExplainer(LR, X_trn, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_tst)
shap.summary_plot(shap_values, X_tst)
shap.summary_plot(shap_values, X_trn, plot_type="bar")

#%%
X = df.values
y = torch.tensor(target).float()

import sklearn
X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

