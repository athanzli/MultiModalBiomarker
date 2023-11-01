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
import torch.functional as F
import lifelines

#
import utils
from model import MMBL

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_random_seed(42)


def att_cat(A, cat_mask):
    r""" Get the average or median attention scores matrix for a particular category.
    
    """
    A_avg = np.median(A[cat_mask], axis=0)
    feature_attn = np.mean(A_avg, axis=0)
    return A_avg, feature_attn

def plot_ft_imp(ft_names, ft_imp, n_top=10, title=None):
    sorted_idx = sorted(range(len(ft_imp)), key=lambda k: ft_imp[k], reverse=True)
    sorted_ft = [ft_names[i] for i in sorted_idx]
    ft_imp = [ft_imp[i] for i in sorted_idx]

    plt.figure(figsize=(10, 3))
    bars = plt.bar(range(n_top), ft_imp[:n_top], align='center')
    plt.xticks(range(n_top), sorted_ft[:n_top], rotation='vertical')

    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.title(title)
    plt.show()

    return sorted_ft

def two_sets_stats(set1, set2):
    intersection = set(set1) & set(set2)
    # union = set(set1) | set(set2)
    only_set1 = set(set1) - intersection
    only_set2 = set(set2) - intersection
    print("Intersection:", len(intersection), '. ', intersection)
    print("Only in set1:", len(only_set1), '. ', only_set1)
    print("Only in set2:", len(only_set2), '. ', only_set2)
    return intersection, only_set1, only_set2

def remove_dashdash(df):
    # TODO just a temp func
    idx=[]
    for i in range(df.shape[1]):
        if (df.iloc[:, i].values=="'--").sum() == df.shape[0]:
            idx.append(i)
    df = df.drop(columns=df.columns[idx])
    return df


#%% ################################# data: mRNA raw data
mrna = pd.read_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts.csv', index_col=0)
# for DEG
tumor_type_codes = [int(barcode.split('-')[3][:-1]) for barcode in mrna.columns.values] # https://docs.gdc.cancer.gov/Encyclopedia/pages/images/TCGA-TCGAbarcode-080518-1750-4378.pdf
sampletype_label = pd.DataFrame(
    index=mrna.columns.values, 
    columns=['sample_type'], 
    data=pd.Series(tumor_type_codes).map({1: 'tumor', 2: 'tumor', 11:'normal'}).values
 )
sampletype_label.to_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_sample_type.csv')
deg = pd.read_csv('./data/TCGA-LIHC_from_GDC_raw/DEG_trn.csv', index_col=0)
deg_set = deg[(deg['padj']<0.05) & (deg['log2FoldChange'].abs()>2)].index # TODO

X = mrna.T
# # remove features that have 0 values across more than 20% samples
# X = X.iloc[:, ~((X == 0).sum(axis=0)>X.shape[0]*0.2).values]
# log normalization
X = np.log(X+1)
X = X.loc[:, deg_set]

#%% ################################# data: save some labels
clinic0 = pd.read_table('./data/TCGA-LIHC_from_GDC_raw/clinical.cart.2023-10-30/clinical.tsv', index_col=0)
clinic0 = remove_dashdash(clinic0)
clinic = clinic0[['case_submitter_id', 'project_id', 'age_at_index', 'days_to_death', 'ajcc_pathologic_stage',
            'vital_status', 'days_to_last_follow_up', 'treatment_or_therapy', 'treatment_type']] # only these columns are useful # the reason there are duplicated samples is that each sample has two rows for treatment info
#%% treatment label
treatment_label = pd.DataFrame(index=clinic['case_submitter_id'].unique(), columns=['treatment'])
for pt in clinic['case_submitter_id'].unique():
    tmp = clinic[clinic['case_submitter_id']==pt][['treatment_or_therapy', 'treatment_type']]
    # 4 cases
    # Pharmaceutical Therapy, NOS |  Radiation Therapy, NOS | treatment
    # No                          |  No                     |  
    # No                          |  Yes                    | 
    # Yes                         |  No                     | 
    # Yes                         |  Yes                    | 
    # not reported                |  not reported           | nan
    if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'no') \
        & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'no'):
        treatment_label.loc[pt, 'treatment'] = 'no treatment'
    if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'no') \
        & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes'):
        treatment_label.loc[pt, 'treatment'] = 'radi'
    if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes') \
        & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'no'):
        treatment_label.loc[pt, 'treatment'] = 'pharma'
    if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes') \
        & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes'):
        treatment_label.loc[pt, 'treatment'] = 'pharma + radi'
    if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'not reported') \
        & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'not reported'):
        treatment_label.loc[pt, 'treatment'] = np.nan

#%% stage label
tmp = clinic[['case_submitter_id', 'ajcc_pathologic_stage']].drop_duplicates()
stage_label = pd.DataFrame(
    index=tmp['case_submitter_id'].values, 
    columns=['pathological_stages'], 
    data=tmp['ajcc_pathologic_stage'].values)
stage_label.loc[(stage_label=="'--").values.flatten(), 'pathological_stages'] = np.nan 
#%% ################################# data: survival
# use days_to_death, days_to_last_follow_up, vital status
# T:
#   for dead patients, T = days_to_death
#   for alive patients, T = days_to_last_follow_up
#   if not reported, then T is NA
# Censor: 
#   if alive, then Censor is labeled as 0
#   if dead, then Censor is labeled as 1
#   if not reported, then Censor is NA

surv = clinic.drop(columns=['treatment_or_therapy', 'treatment_type'])
surv = surv.drop_duplicates()
surv.index = surv['case_submitter_id'].values
surv = surv.drop(columns=['case_submitter_id'])
surv.loc[surv['vital_status']=='Dead', 'T'] = surv.loc[surv['vital_status']=='Dead', 'days_to_death']
surv.loc[surv['vital_status']=='Alive', 'T'] = surv.loc[surv['vital_status']=='Alive', 'days_to_last_follow_up']
surv = surv.dropna(subset=['T'])
surv = surv[~(surv['T']=="'--")]
surv['T'] = surv['T'].astype(float)
surv['E'] = surv['vital_status'].map({'Dead': 1, 'Alive': 0})

# TODO CHOOSE
# surv['group'] = treatment_label.loc[surv.index] # TODO choose label group
# surv['group'] = surv['group'].map({'no treatment': 'no treatment', 'radi': 'treatment', 'pharma': 'treatment', 'pharma + radi': 'treatment'})

surv['group'] = stage_label.loc[surv.index] # TODO choose label group
surv['group'] = surv['group'].map({
    'Stage I': 'Stage I',
    'Stage II': 'Stage II',
    'Stage III': 'Stage III',
    'Stage IIIA': 'Stage III',
    'Stage IIIB': 'Stage III',
    'Stage IIIC': 'Stage III',
    'Stage IVA': 'Stage IV',
    'Stage IVB': 'Stage IV',
    'Stage IV': 'Stage IV',
})
print(surv['group'].value_counts())

#%%
df = surv
ax = plt.subplot(111)
kmf = lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter()
for name, grouped_df in df.groupby('group'):
    kmf.fit(grouped_df["T"], grouped_df["E"], label=name)
    kmf.plot_survival_function(ax=ax)
mask = (df['group'] == 'Stage II') | ((df['group'] == 'Stage III')) # TODO choose
p = lifelines.statistics.pairwise_logrank_test(
    df['T'][mask], df['group'][mask], df['E'][mask]).summary['p']
ax.text(0.3, 0.85, f'log-rank test p = {p.values[0]:.2e}', transform=ax.transAxes)

#%% #
# from lifelines import CoxPHFitter
# from lifelines.datasets import load_regression_dataset
# regression_dataset = load_regression_dataset() # a Pandas DataFrame
# cph = CoxPHFitter()
# cph.fit(regression_dataset, 'T', event_col='E')
# cph.print_summary()

# pt = regression_dataset.loc[0]
# cph.predict_survival_function(pt).rename(columns={0:'CoxPHFitter'}).plot()

#%% ################################# some initial survival analysis

#%% ################################# prepare X and y
pts = np.unique(['-'.join(pt.split('-')[:3]) for pt in X.index])
# choose only tumor samples
X = X.loc[pd.Series(tumor_type_codes).map({1: True, 2: False, 11:False}).values, :] # NOTE here map 2 to false, because clinic's patients are all with label 1
y = stage

#%% ########################### split and prep data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(sss.split(X, y))
X_trn, X_tst = X.iloc[train_idx], X.iloc[test_idx]
y_trn, y_tst = y[train_idx], y[test_idx]
print(np.unique(y_trn, return_counts=True))
X_trn.T.to_csv("./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_trn.csv")
pd.DataFrame(index=X_trn.index.values, columns=['sample_type'], data=y_trn).to_csv('./data/TCGA-LIHC_from_GDC_raw/TCGA_LIHC_mRNA_raw_counts_sample_type_trn.csv')

X_trn = X_trn
X_tst = X_tst
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

# TODO DEL The data from UCSC
# clinic = pd.read_table("C:/Users/athan/OneDrive/Desktop/MMBiomarker/MMBiomarker/data/TCGA-LIHC_from_UCSC/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix")
# tumor_type_info = clinic.groupby(['_PATIENT', 'sample_type']).size().unstack()
# (~tumor_type_info['Primary Tumor'].isna()).sum()
# (~tumor_type_info['Solid Tissue Normal'].isna()).sum()
# (~tumor_type_info['Recurrent Tumor'].isna()).sum()
# mrna = pd.read_talbe('C:\Users\athan\OneDrive\Desktop\MMBiomarker\MMBiomarker\data\TCGA-LIHC_from_UCSC\HiSeqV2')




#%% ###########################################################################
# ################################# model #####################################
# #############################################################################
model = MMBL(
    seq_len = X_trn.shape[1],
    emb_dim = 32,
    mlp_channels = [X_trn.shape[1], X_trn.shape[1], 1],
    n_heads = 1,
    n_encoders = 1,
)

#%% ############################### train
# TODO  change accordingly
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

#%% ############################### training for binary/multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = torch.nn.CrossEntropyLoss()

n_epochs = 30
for epoch in range(n_epochs):
    model.train()

    # # TODO batch training
    # epoch_loss = 0
    # for batch in trn_loader:
    #     x, y = batch
    #     # TODO change accordingly
    #     logits = model(x)
    #     loss = loss_fn(logits.squeeze(), y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     epoch_loss += loss.item()
    # epoch_loss /= len(trn_loader)
    # trn_losses.append(epoch_loss)

    # TODO whole batch training
    logits = model(X_trn)
    loss = loss_fn(logits.squeeze(), y_trn)
    trn_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO change accordingly
    model.eval()
    logits = model(X_trn).detach()
    acc = (logits.sigmoid().round() == y_trn.unsqueeze(1)).float().mean()
    print(f'epoch {epoch}, loss {loss.item()}, trn accuracy {acc.item()}')
    # print(f'epoch {epoch}, loss {loss.item()}')
    trn_acc.append(acc.item())


    # TODO change accordingly
    model.eval()
    # logits = model(X_tst).detach()
    logits = model(X_tst).detach().squeeze()
    # loss = loss_fn(logits.squeeze(), y_tst)
    # tst_losses.append(loss.item())
    acc = (logits.sigmoid().round() == y_tst).float().mean()
    print(f'test accuracy {acc.item():.3f}')
    tst_acc.append(acc.item())


#%% ############################### training for hazard function regression (survival)
def hazard_loss(

)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

n_epochs = 30
for epoch in range(n_epochs):
    model.train()

    # # TODO batch training
    # epoch_loss = 0
    # for batch in trn_loader:
    #     x, y = batch
    #     # TODO change accordingly
    #     logits = model(x)
    #     loss = loss_fn(logits.squeeze(), y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     epoch_loss += loss.item()
    # epoch_loss /= len(trn_loader)
    # trn_losses.append(epoch_loss)

    # TODO whole batch training
    logits = model(X_trn)
    loss = loss_fn(logits.squeeze(), y_trn)
    trn_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO change accordingly
    model.eval()
    logits = model(X_trn).detach()
    acc = (logits.sigmoid().round() == y_trn.unsqueeze(1)).float().mean()
    print(f'epoch {epoch}, loss {loss.item()}, trn accuracy {acc.item()}')
    # print(f'epoch {epoch}, loss {loss.item()}')
    trn_acc.append(acc.item())


    # TODO change accordingly
    model.eval()
    # logits = model(X_tst).detach()
    logits = model(X_tst).detach().squeeze()
    # loss = loss_fn(logits.squeeze(), y_tst)
    # tst_losses.append(loss.item())
    acc = (logits.sigmoid().round() == y_tst).float().mean()
    print(f'test accuracy {acc.item():.3f}')
    tst_acc.append(acc.item())



#%% plot
_, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(trn_losses, label='train')
# axes[0].plot(tst_losses, label='val')
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
print('Test accuracy', tst_acc[-1])

#%% #################################### analysis
## See biomarker genes
liver_gs_G = pd.read_table('./data/GoldStandardList/C0024620_disease_gda_summary.tsv')
liver_gs_G = liver_gs_G['Gene'].values

#%% get effective attention
model.eval()
A_eff = model.module1.effective_attention(X_trn).detach().numpy()
#%% plot effective attention
n_top = 60
A_avg, ft_imp = att_cat(A_eff, y_trn==1)
sorted_ft = plot_ft_imp(ft_names, ft_imp, n_top=n_top) # show only n_top
argsorted = np.argsort(ft_imp)[::-1]
two_sets_stats(sorted_ft[:n_top], liver_gs_G)
sns.heatmap(A_avg, cmap='Reds', ); plt.show()

A_avg, ft_imp = att_cat(A_eff, y_trn==0)
sorted_ft = plot_ft_imp(ft_names, ft_imp, n_top=n_top)
two_sets_stats(sorted_ft[:n_top], liver_gs_G)
sns.heatmap(A_avg, cmap='Reds', ); plt.show()

#%% analyse attention scores


#%% #################################### backup code
