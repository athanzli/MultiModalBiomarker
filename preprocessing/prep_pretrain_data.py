# import pandas as pd
# import numpy as np

# DATA_PATH = '/home/che82/athan/MMBiomarker_data'

# X1 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/mRNA_deg_0.05_1.csv", index_col=0)
# X2 = pd.read_csv(DATA_PATH + "/TCGA-PanCancerData-preprocessed/miRNA.csv", index_col=0)
# samples = np.intersect1d(X1.index, X2.index)
# X1 = X1.loc[samples, :]
# X2 = X2.loc[samples, :]