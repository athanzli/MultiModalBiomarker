#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import lifelines

ALL_TCGAPROJ = [
    'TCGA-GBM', 'TCGA-OV', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-PRAD',
    'TCGA-BLCA', 'TCGA-TGCT', 'TCGA-ESCA', 'TCGA-PAAD', 'TCGA-KIRP',
    'TCGA-LIHC', 'TCGA-CESC', 'TCGA-SARC', 'TCGA-BRCA', 'TCGA-THYM',
    'TCGA-MESO', 'TCGA-COAD', 'TCGA-STAD', 'TCGA-SKCM', 'TCGA-CHOL',
    'TCGA-KIRC', 'TCGA-THCA', 'TCGA-HNSC', 'TCGA-LAML', 'TCGA-READ',
    'TCGA-LGG', 'TCGA-DLBC', 'TCGA-KICH', 'TCGA-UCS', 'TCGA-ACC',
    'TCGA-PCPG', 'TCGA-UVM', 'TCGA-UCEC']

DATA_PATH = '/home/che82/athan/MMBiomarker_data' # DATA_PATH = "C:/Users/athan/OneDrive/Desktop/MMBiomarker_data"

#%%
# load sample data
ss1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-raw/sample_sheet_mrna.csv", index_col=0)
fileid = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-intermediate/fileids_mRNA.csv", index_col=0)
st1 = ss1.loc[fileid.values.flatten(), ['Sample ID', 'Sample Type']]
ss2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-raw/sample_sheet_mirna.csv", index_col=0)
fileid = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-intermediate/fileids_miRNA.csv", index_col=0)
st2 = ss2.loc[fileid.values.flatten(), ['Sample ID', 'Sample Type']]
## sanity check
stt1 = st1['Sample ID'] + '-' + st1['Sample Type']
stt2 = st2['Sample ID'] + '-' + st2['Sample Type']
assert np.intersect1d(stt1, stt2).shape[0] == np.intersect1d(st1['Sample ID'], st2['Sample ID']).shape[0]
##
st1.index = st1['Sample ID']; st1.drop(columns='Sample ID', inplace=True)
st2.index = st2['Sample ID']; st2.drop(columns='Sample ID', inplace=True)

# load clinical data
clic = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/clinic.csv", index_col=0)

#%% load omics data
## choose overlapped samples
X1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/mRNA_deg_0.05_1.csv", index_col=0)
X2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/miRNA.csv", index_col=0)
samples = np.intersect1d(X1.index, X2.index)
X1 = X1.loc[samples, :]
X2 = X2.loc[samples, :]
assert np.setdiff1d(samples, st1.index).shape[0] == 0
st = st1.loc[samples, :]
X = pd.concat([X1, X2], axis=1)

## TODO maybe add clinical modality

#%% ###
all_samples = np.union1d(X1.index, X2.index)
sel_pt = ['-'.join(all_samples[i].split('-')[:-1]) for i in range(len(all_samples))]
tmp = clic.loc[sel_pt, 'project_id'].to_frame()
tmp['patient'] = tmp.index
tmp = tmp.drop_duplicates()
assert not (tmp.groupby(['patient', 'project_id']).size().values > 1).any(), "Existing patients belong to multiple projects."
tmp['project_id'].to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project-label-for-patient.csv')

tmp = clic.loc[sel_pt, 'project_id'].to_frame()
tmp.index = all_samples
tmp['sample'] = tmp.index
tmp = tmp.drop_duplicates()
assert not (tmp.groupby(['sample', 'project_id']).size().values > 1).any(), "Existing samples belong to multiple projects."
tmp['project_id'].to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project-label-for-sample.csv')

#%% 
###############################################################################
# TCGA proj classification (33 cancer typing)
###############################################################################
# currently just the 33 TCGA projects id classification TODO later consider primary diagnosis?
# clic['primary_diagnosis'].value_counts().index.values
sel_pt = ['-'.join(X.index[i].split('-')[:-1]) for i in range(X.shape[0])]
## plot distribution
### 33 TCGA projects
fig = plt.figure(figsize=(10, 5))
sns.barplot(
    y=clic.loc[sel_pt, 'project_id'].value_counts().values,
    x=clic.loc[sel_pt, 'project_id'].value_counts().index,
)
plt.xticks(rotation=90)
plt.ylabel('Number of samples')
plt.show()
fig = plt.figure(figsize=(10, 5))
sns.barplot(
    y=clic['project_id'].value_counts().values,
    x=clic['project_id'].value_counts().index,
)
plt.xticks(rotation=90)
plt.ylabel('Number of patients')
plt.show()
##
y_proj = pd.factorize(clic.loc[sel_pt, 'project_id'])[0]
uniques = clic.loc[sel_pt, 'project_id'].unique()
mapping = {name: i for i, name in enumerate(uniques)}
assert np.all((mapping[clic['project_id'].values[i]] == y_proj[i]) for i in range(y_proj.shape[0]))

X1.to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project/mRNA.csv')
X2.to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project/miRNA.csv')
np.save(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project/y.npy', y_proj)
pd.DataFrame(index=mapping.keys(), data=mapping.values(), columns=['level']) \
    .to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/TCGA-project/y_mapping.csv')

#%% 
###############################################################################
# metastasis prediction (binary classification of primary tumor vs. metastasis)
###############################################################################
st_mts = st.copy()
print(st_mts['Sample Type'].value_counts())
st_mts = st_mts[(st_mts['Sample Type'] == 'Primary Tumor') | (st_mts['Sample Type'] == 'Metastatic')]
y_metas = pd.factorize(st_mts['Sample Type'])[0] # metastasis is 1
X1.loc[st_mts.index].to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/metastasis/mRNA.csv')
X2.loc[st_mts.index].to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/metastasis/miRNA.csv')
np.save(DATA_PATH + '/TCGA-PanCancerData-preprocessed/metastasis/y.npy', y_metas)

#%% 
###############################################################################
# survival
###############################################################################
"""use days_to_death, days_to_last_follow_up, vital status
T:
  for dead patients, T = days_to_death
  for alive patients, T = days_to_last_follow_up
  if not reported, then T is NA
Censor: 
  if alive, then Censor is labeled as 0
  if dead, then Censor is labeled as 1
  if not reported, then Censor is NA

TODO ref to see if you missed any details https://github.com/luisvalesilva/multisurv/blob/master/data/preprocess_clinical.ipynb
"""
# prepare data
clic.loc[clic['vital_status']=='Dead', 'T'] = clic.loc[clic['vital_status']=='Dead', 'days_to_death']
clic.loc[clic['vital_status']=='Alive', 'T'] = clic.loc[clic['vital_status']=='Alive', 'days_to_last_follow_up']
clic.loc[clic['T']=="'--", 'T'] = np.nan
clic['T'] = clic['T'].astype(float)
clic['E'] = clic['vital_status'].map({'Dead': 1, 'Alive': 0})
clic.loc[clic['E']=="'--", 'E'] = np.nan
assert not (clic[clic['T'].isna()]['E'].isna().all())
assert clic[clic['E'].isna()]['T'].isna().all()
sv = clic[['T', 'E']][clic['T'].notna()]
assert sv['T'].isna().sum() + sv['E'].isna().sum() == 0
sv = sv[sv['T'] >= 0]
sv['E'] = sv['E'].astype(int)

assert (X1.index == X2.index).all() & (X1.index == X.index).all()
Xm = X.copy()
Xm = Xm.loc[st[(st['Sample Type'] != 'Solid Tissue Normal')].index]
st = st.loc[st[(st['Sample Type'] != 'Solid Tissue Normal')].index]
sel_pt = ['-'.join(Xm.index[i].split('-')[:-1]) for i in range(Xm.shape[0])]
Xm.index = sel_pt
Xm = Xm.groupby(Xm.index).mean() # aggregate
pts = np.intersect1d(Xm.index, sv.index)
Xm = Xm.loc[pts]
sv = sv.loc[pts]

#
sns.histplot(x=sv['T'], hue=sv['E'], bins=100)

# save
Xm.iloc[:, :X1.shape[1]].to_csv(
    DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival/mRNA.csv')
Xm.iloc[:, :X2.shape[1]].to_csv(
    DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival/miRNA.csv')
sv.to_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival/survival.csv')

#%% 
###############################################################################
# stage classification
###############################################################################

# def print_stage_info(project):
#     print("Project: ", project)
#     mask = clic['project_id'] == project
#     print(clic[mask]['ajcc_pathologic_stage'].value_counts())
#     print()

# for proj in clic['project_id'].unique():
#     print_stage_info('proj')

""" from 'figo_stage' alone
'Stage IA1', 'Stage IA2', 'Stage IB1', 'Stage IB2', 'Stage IC',
       'Stage IIA1', 'Stage IIA2', 'Stage IIIC1', 'Stage IIIC2'

The following 4 cancer types have 'figo_stage' but not 'ajcc_pathologic_stage':
'TCGA-OV', 'TCGA-CESC', 'TCGA-UCS', 'TCGA-UCEC'
"""
sg = clic.copy()
sg.loc[sv.index, 'T'] = sv['T']
sg.loc[sv.index, 'E'] = sv['E']
sg['ajcc_pathologic_and_figo_stage'] = sg['ajcc_pathologic_stage']
mask = (sg['figo_stage']!="'--") # & sg['figo_stage'].notna()
sg.loc[sg[mask].index, 'ajcc_pathologic_and_figo_stage'] = sg.loc[mask, 'figo_stage']
print(sg['ajcc_pathologic_and_figo_stage'].value_counts())
early_stage = [
    'Stage 0', 'Stage I', 'Stage IA', 'Stage IB', 'Stage IA1', 'Stage IA2', 
    'Stage IB1', 'Stage IB2', 'Stage IC', 'Stage IS', 'Stage II', 'Stage IIA', 
    'Stage IIB', 'Stage IIC', 'Stage IIA1', 'Stage IIA2'
]
middle_stage = [
    'Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC', 'Stage IIIC1', 'Stage IIIC2'
]
late_stage = [
    'Stage IV', 'Stage IVA', 'Stage IVB', 'Stage IVC'
]
sg['broad_stage'] = sg['ajcc_pathologic_and_figo_stage'].map({
    **{s: 'Early' for s in early_stage}, 
    **{s: 'Middle' for s in middle_stage}, 
    **{s: 'Late' for s in late_stage},
})
sg = sg[sg['broad_stage'].notna()]
print(sg['broad_stage'].value_counts())

## plot KM curve
df = sg[sg['broad_stage'].notna() & (sg['T'].notna()) & (sg['E'].notna())]
ax = plt.subplot(111)
kmf = lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter()
for name, grouped_df in df.groupby('broad_stage'):
    kmf.fit(grouped_df["T"], grouped_df["E"], label=name)
    kmf.plot_survival_function(ax=ax)
## test                     # TODO Late
mask = (df['broad_stage'] == 'Early') | ((df['broad_stage'] == 'Middle')) # TODO choose
p = lifelines.statistics.pairwise_logrank_test(
    df['T'][mask], df['broad_stage'][mask], df['E'][mask]).summary['p']
ax.text(0.3, 0.85, f'log-rank test p = {p.values[0]:.2e}', transform=ax.transAxes)

##
assert (st.index==X.index).all()
Xm = X[st['Sample Type']!='Solid Tissue Normal'].copy()
Xm.index = ['-'.join(Xm.index[i].split('-')[:-1]) for i in range(Xm.shape[0])]
Xm = Xm[Xm.index.isin(sg.index)]
Xm = Xm.groupby(Xm.index).mean() # aggregate
y = sg.loc[Xm.index]['broad_stage']
assert (y.index==Xm.index).all()
Xm.iloc[:, :X1.shape[1]].to_csv(
    DATA_PATH + '/TCGA-PanCancerData-preprocessed/stage-classification/mRNA.csv')
Xm.iloc[:, :X2.shape[1]].to_csv(
    DATA_PATH + '/TCGA-PanCancerData-preprocessed/stage-classification/miRNA.csv')
np.save(
    DATA_PATH + '/TCGA-PanCancerData-preprocessed/stage-classification/y.npy', 
    y.map({'Early': 0, 'Middle': 1, 'Late': 2}).values)

#%% 
###############################################################################
# cancer subtyping
###############################################################################


#%% caogao
deg = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/mRNA_deg_res.csv", index_col=0)
for g in ['TP53', 'KRAS', 'EGFR', 'BRAF', 'PTEN', 'PIK3CA', 'APC', 'NF1', 
          'RB1', 'CDKN2A', 'ALK']:
    print(f"LFC: {deg.loc[g]['log2FoldChange']}", f"padj: {deg.loc[g]['padj']}")



# #%% 
# ###############################################################################
# # survival binary (TEMP)
# ###############################################################################
# sv = pd.read_csv(DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival/survival.csv', index_col=0)
# sv = sv[~sv.index.isin(sv[(sv['T'] < sv['T'].median()) & (sv['E'] == 0)].index)]
# sv['binary'] = (sv['T'] < sv['T'].median()).values
# sv['binary'] = sv['binary'].map({True: 0, False: 1}) # 0: early, 1: late

# assert (X1.index == X2.index).all() & (X1.index == X.index).all()
# Xm = X.copy()
# Xm = Xm.loc[st[(st['Sample Type'] != 'Solid Tissue Normal')].index]
# st = st.loc[st[(st['Sample Type'] != 'Solid Tissue Normal')].index]
# sel_pt = ['-'.join(Xm.index[i].split('-')[:-1]) for i in range(Xm.shape[0])]
# Xm.index = sel_pt
# Xm = Xm.groupby(Xm.index).mean() # aggregate
# pts = np.intersect1d(Xm.index, sv.index)
# Xm = Xm.loc[pts]
# sv = sv.loc[pts]

# # save
# Xm.iloc[:, :X1.shape[1]].to_csv(
#     DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival-binary/mRNA.csv')
# Xm.iloc[:, :X2.shape[1]].to_csv(
#     DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival-binary/miRNA.csv')
# np.save(
#     DATA_PATH + '/TCGA-PanCancerData-preprocessed/survival-binary/survival_binary.csv',
#     sv['binary'].values
# )
