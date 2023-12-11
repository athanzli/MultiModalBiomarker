#%%
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from sklearn.model_selection import StratifiedShuffleSplit
#%% load data
DATA_PATH = '/home/che82/athan/MMBiomarker_data' # DATA_PATH = "C:/Users/athan/OneDrive/Desktop/MMBiomarker_data"
#%% preprocess clinic data
def preprocess_clinic_data(clic):
    # only these columns are useful 
    # the reason there are duplicated samples is that 
    #   each sample has two rows for treatment info
    """
    
    """
    # TODO
    clic = clic[[
        'case_submitter_id', 'project_id',
        'age_at_index',  'days_to_death', 
        'primary_diagnosis',
        'ajcc_pathologic_stage',
        'figo_stage',
        'vital_status', 'days_to_last_follow_up', 'days_to_diagnosis', 
        'treatment_or_therapy', 'treatment_type',
        'site_of_resection_or_biopsy', 'tissue_or_organ_of_origin',
        'morphology',
        'age_at_diagnosis', 'gender', 'ethnicity', 'race', # TODO ethnicity and race can be consolidated, see https://github.com/luisvalesilva/multisurv/blob/master/data/preprocess_clinical.ipynb
        'prior_malignancy', # had cancer previously 
        'year_of_diagnosis'
    ]]

    treatment_label = pd.DataFrame(index=clic['case_submitter_id'].unique(), columns=['treatment'])
    for sp in clic['case_submitter_id'].unique():
        tmp = clic[clic['case_submitter_id']==sp][['treatment_or_therapy', 'treatment_type']]
        # 5 cases
        # Pharmaceutical Therapy, NOS |  Radiation Therapy, NOS | treatment
        # No                          |  No                     |  
        # No                          |  Yes                    | 
        # Yes                         |  No                     | 
        # Yes                         |  Yes                    | 
        # not reported                |  not reported           | nan
        # '--'                        |  '--'                   | '--'
        if (tmp['treatment_type'].values[0] == "'--" ) & (tmp['treatment_or_therapy'].values[0] == "'--"):
            treatment_label.loc[sp, 'treatment'] = np.nan
            continue    
        if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'no') \
            & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'no'):
            treatment_label.loc[sp, 'treatment'] = 'no treatment'
        if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'no') \
            & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes'):
            treatment_label.loc[sp, 'treatment'] = 'radi'
        if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes') \
            & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'no'):
            treatment_label.loc[sp, 'treatment'] = 'pharma'
        if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes') \
            & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'yes'):
            treatment_label.loc[sp, 'treatment'] = 'pharma + radi'
        if (tmp[tmp['treatment_type']=='Pharmaceutical Therapy, NOS']['treatment_or_therapy'].values[0] == 'not reported') \
            & (tmp[tmp['treatment_type']=='Radiation Therapy, NOS']['treatment_or_therapy'].values[0] == 'not reported'):
            treatment_label.loc[sp, 'treatment'] = np.nan
  
    clic.index = clic['case_submitter_id'].values
    clic = clic.drop(columns=['treatment_type', 'treatment_or_therapy'])
    clic = clic.drop_duplicates()
    clic['treatment'] = treatment_label.loc[clic.index, 'treatment'].values

    return clic


clic1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-raw/clinical_mrna.csv", index_col=0)
clic2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-raw/clinical_mirna.csv", index_col=0)
clic1 = preprocess_clinic_data(clic1)
clic2 = preprocess_clinic_data(clic2)
## choose overlapped patients
###
# sel = np.intersect1d(clic1.index, clic2.index)
# clic1 = clic1.loc[sel, :]
# clic2 = clic2.loc[sel, :]
# clic1.eq(clic2).all()
# all(clic1.iloc[np.where(clic1.treatment != clic2.treatment)[0]]['treatment'].isna())
# clic = clic1
# ### overlapped patients between clinic and mRNA/miRNA
# pt = np.unique(['-'.join(mrna.index[i].split('-')[:-1]) for i in range(mrna.shape[0])])
# clic = clic.loc[pt]

## choose union
assert np.setdiff1d(clic1.index, clic2.index).shape[0] == 0
clic = clic2
## save data
clic.to_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/clinic.csv")

#%% backup
# def filter_lognorm(data, minthres=0, size_factor=1e6):
#     data = data[data.gt(0).sum(axis=1) > minthres * data.shape[1]]
#     data = np.log2(((data / data.sum(axis=0)) * size_factor) + 1)
#     return data
# mrna = filter_lognorm(mrna) # DEL as did scBERT, RPM and log-transform is a valid preprocessing method for mRNA
# mirna = filter_lognorm(mirna) # DEL RPM and log-transform is a valid preprocessing method for miRNA, as shown in paper ..

#%% caogao
# bs = pd.read_table(DATA_PATH + "/TCGA-PanCancerData-biospecimen/aliquot.tsv", index_col=0)
# bs = pd.read_table(DATA_PATH + "/TCGA-PanCancerData-biospecimen/analyte.tsv", index_col=0)
# bs = pd.read_table(DATA_PATH + "/TCGA-PanCancerData-biospecimen/portion.tsv", index_col=0)
# bs = pd.read_table(DATA_PATH + "/TCGA-PanCancerData-biospecimen/sample.tsv", index_col=0)
# bs = pd.read_table(DATA_PATH + "/TCGA-PanCancerData-biospecimen/slide.tsv", index_col=0)
# bs.columns.values
# bs['tumor_descriptor'].value_counts()
