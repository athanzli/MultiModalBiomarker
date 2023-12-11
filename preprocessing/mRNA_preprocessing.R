setwd('/home/che82/athan/MMBiomarker_data/')

data <- read.csv('./TCGA-PanCancerData-intermediate/mRNA.csv', row.names=1)
colnames(data) <- gsub("\\.", "-", colnames(data))
# remove zero features
data <- data[rowSums(data)>0, ]

# DEG
library(DESeq2)
library(dplyr)
## data: rows are genes, cols are samples
ss <- read.csv("./TCGA-PanCancerData-raw/sample_sheet_mrna.csv", header=TRUE, row.names=1)
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

colData <- data.frame(row.names=colnames(data),
                      condition=y)
colData$condition <- factor(colData$condition, levels = c('normal', 'tumor'))

dds <- DESeqDataSetFromMatrix(
  countData = data,
  colData = colData,
  design = ~condition
)
dds <- DESeq(dds)

# get res
res <- results(dds, contrast = c('condition', 'tumor', 'normal'))
res <- res[order(res$padj),]
res <- na.omit(res)
deg <- res[(res$padj < 0.05) & (abs(res$log2FoldChange) > 1), ] # TODO the cutoff

write.csv(res, './TCGA-PanCancerData-preprocessed/mRNA_deg_res.csv')

#
column_sums <- colSums(data)
data <- sweep(data, 2, column_sums, "/") * 1e6
data <- log2(data + 1)
data <- t(data)

newdata <- data[y=='tumor', rownames(deg)]
write.csv(newdata, './TCGA-PanCancerData-preprocessed/mRNA_deg_0.05_1.csv')

###############################################################################
# old version
###############################################################################
# " DEGs from mRNA raw counts data matrix.

# "

# library(DESeq2)
# library(edgeR)
# setwd('C:/Users/athan/OneDrive/Desktop/MMBiomarker/data/TCGA-LIHC_from_GDC_raw/')

# # NOTE OPTION
# # count_matrix <- read.csv("TCGA_LIHC_mRNA_raw_counts_trn.csv", row.names=1) # rows are genes, cols are samples
# # sample_info <- read.csv("TCGA_LIHC_mRNA_raw_counts_sample_type_trn.csv", header=TRUE, row.names=1)
# count_matrix <- read.csv("mRNA_raw_counts_trn.csv", row.names=1) # rows are genes, cols are samples
# sample_info <- read.csv("TCGA_LIHC_mRNA_raw_counts_sample_type_trn.csv", header=TRUE, row.names=1)

# colnames(count_matrix) <- lapply(colnames(count_matrix), function(x) gsub('\\.', '-', x)) # NOTE
# all(rownames(sample_info)==colnames(count_matrix))

# colData <- data.frame(row.names=rownames(sample_info),
#                       condition=sample_info$sample_type)
# colData$condition <- factor(colData$condition, levels = c('normal', 'tumor'))

# dds <- DESeqDataSetFromMatrix(
#   countData = count_matrix,
#   colData = colData,
#   design = ~condition
# )
# dds <- DESeq(dds)

# # get res
# res <- results(dds, contrast = c('condition', 'tumor', 'normal'))
# res <- res[order(res$pvalue),]
# res <- na.omit(res)
# # DEG <- res[(res$padj < 0.05) & (abs(res$log2FoldChange) > 2), ] # TODO the cutoff
# # dim(DEG)

# write.csv(res, 'res_trn.csv')
