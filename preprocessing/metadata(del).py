""" NOTE s
case_id: cases (patients)
case_submitter_id: e.g. TCGA-FA-A7Q1. 1-to-1 mapping to case_id

"""

#%%
import pandas as pd
import numpy as np

sample_sheet1 = pd.read_csv("../data/TCGA-PanCancerData-raw/sample_sheet_mrna.csv", index_col=0)
sample_sheet2 = pd.read_csv("../data/TCGA-PanCancerData-raw/sample_sheet_mirna.csv", index_col=0)
cli1 = pd.read_table("/home/che82/Downloads/clinical.cart.2023-11-21/clinical.tsv", index_col=0)
cli2 = pd.read_table("/home/che82/Downloads/clinical.cart.2023-11-21(1)/clinical.tsv", index_col=0)

#%%
sample_type = sample_sheet1[['Sample ID', 'Sample Type']].drop_duplicates()
sample_type.index = sample_type['Sample ID']
sample_type.drop(columns='Sample ID', inplace=True)
print(sample_type.value_counts())
sample_type.to_csv("../data/TCGA-PanCancerData-intermediate/sample_type.csv")

#%%
####
cli = pd.concat([cli, cli1, cli2, cli3])

cli.columns.values

cli['case_submitter_id'].value_counts()

sample_sheet['Case ID'].value_counts()

cli['case_submitter_id'].value_counts().index

len(cli['project_id'].value_counts().index)

cli['case_submitter_id'].value_counts().index
sample_sheet['Case ID'].value_counts().index

np.setdiff1d(
    cli['case_submitter_id'].value_counts().index,
    sample_sheet['Case ID'].value_counts().index)

sample_sheet_mirna

np.intersect1d(
    np.unique((sample_sheet.index)).astype(str),
    cli['case_id'].value_counts().index.values.astype(str)
)

sample_sheet['Sample Type'].value_counts()



np.unique((np.concatenate([np.unique(cli.index), np.unique(sample_sheet2.index)]))).shape
sample_sheet1.shape[0] + sample_sheet2.shape[0]


np.unique(cli1.index).shape

cur=["TCGA-LUSC", "TCGA-KIRC", "TCGA-THCA", "TCGA-ESCA", "TCGA-STAD", "TCGA-LGG", "TCGA-LIHC", "TCGA-GBM", "TCGA-HNSC", "TCGA-UCEC", "TCGA-THYM", "TCGA-COAD", "TCGA-SARC", "TCGA-KICH", "TCGA-PCPG", "TCGA-READ", "TCGA-LAML", "TCGA-PAAD", "TCGA-KIRP", "TCGA-BLCA", "TCGA-TGCT", "TCGA-CESC", "TCGA-MESO", "TCGA-CHOL", "TCGA-DLBC", "TCGA-ACC", "TCGA-UVM", "TCGA-UCS"]
np.setdiff1d(ALL, cur)


#######
"""
- why certain samples can have multiple files?
- 
"""
sample_sheet1['File Name'].value_counts()
sample_sheet2['File Name'].value_counts()
np.intersect1d(
    sample_sheet1['File Name'].unique(),
    sample_sheet2['File Name'].unique()
).shape

sample_sheet1['Case ID'].value_counts()
sample_sheet2['Case ID'].value_counts()
np.intersect1d(
    sample_sheet1['Case ID'].unique(),
    sample_sheet2['Case ID'].unique()
).shape

sample_sheet1['Sample ID'].value_counts()
sample_sheet2['Sample ID'].value_counts()
np.intersect1d(
    sample_sheet1['Sample ID'].unique(),
    sample_sheet2['Sample ID'].unique()
).shape


cli1['case_']
cli2

#%%
###############################################################################
#
###############################################################################
sample_sheet = sample_sheet2
dup_samples = sample_sheet['Sample ID'].value_counts().index[sample_sheet['Sample ID'].value_counts()>1]
for ds in dup_samples:
    print(sample_sheet[sample_sheet['Sample ID'] == ds])
    print('File name:', sample_sheet[sample_sheet['Sample ID'] == ds]['File Name'].values)
    print('#############################################')
    print('#############################################')

#%%
###############################################################################
#
###############################################################################
import glob
data_paths = sorted(glob.glob('/home/che82/athan/MMBiomarker/data/TCGA-PanCancerData-raw/mRNA/*.tsv'))


#%%
s = pd.read_table('/home/che82/Downloads/biospecimen.cases_selection.2023-11-21/sample.tsv', index_col=0)
s['case_id'].value_counts()


