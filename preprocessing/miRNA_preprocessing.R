setwd('/home/che82/athan/MMBiomarker_data/')

data <- read.csv('./TCGA-PanCancerData-intermediate/miRNA.csv', row.names=1)
colnames(data) <- gsub("\\.", "-", colnames(data))
# remove zero features
data <- data[rowSums(data)>0, ]

# DEG
library(DESeq2)
library(dplyr)
## data: rows are genes, cols are samples
ss <- read.csv("./TCGA-PanCancerData-raw/sample_sheet_mirna.csv", header=TRUE, row.names=1)
sample_type <- c()
for (i in 1:length(colnames(data))) {
  idx <- which(ss$Sample.ID==colnames(data)[i])
  if (length(idx) > 1) {
    for (j in 1:length(idx)) {
      if (j == 1) {
        type <- ss$Sample.Type[idx[j]]  
      }
      stopifnot(ss$Sample.Type[idx[j]] == type)
    }
  }
  sample_type <- c(sample_type, unique(ss$Sample.Type[idx]))
}
print(table(sample_type))
y <- sample_type %>%
  recode("Solid Tissue Normal" = 'normal', .default = 'tumor')

# miRNA already did RPM
data <- log2(data + 1)
data <- t(data)

# newdata <- data[y=='tumor', ] # this will exclude TCGA-GBM
newdata <- data
write.csv(newdata, './TCGA-PanCancerData-preprocessed/miRNA.csv')
