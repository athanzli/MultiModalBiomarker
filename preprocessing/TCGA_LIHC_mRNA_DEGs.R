" DEGs from mRNA raw counts data matrix.

"

library(DESeq2)
library(edgeR)
setwd('C:/Users/athan/OneDrive/Desktop/MMBiomarker/MMBiomarker/data/TCGA-LIHC_from_GDC_raw/')

count_matrix <- read.csv("TCGA_LIHC_mRNA_raw_counts_trn.csv", row.names=1) # rows are genes, cols are samples
sample_info <- read.csv("TCGA_LIHC_mRNA_raw_counts_sample_type_trn.csv", header=TRUE, row.names=1)
colnames(count_matrix) <- lapply(colnames(count_matrix), function(x) gsub('\\.', '-', x))
all(rownames(sample_info)==colnames(count_matrix))

colData <- data.frame(row.names=rownames(sample_info),
                      condition=sample_info$sample_type)
colData$condition <- factor(colData$condition, levels = c('normal', 'tumor'))

dds <- DESeqDataSetFromMatrix(
  countData = count_matrix,
  colData = colData,
  design = ~condition
)
dds <- DESeq(dds)

# get res
res <- results(dds, contrast = c('condition', 'tumor', 'normal'))
res <- res[order(res$pvalue),]
res <- na.omit(res)
# DEG <- res[(res$padj < 0.05) & (abs(res$log2FoldChange) > 2), ] # TODO the cutoff
# dim(DEG)

write.csv(res, 'res_trn.csv')


# #
# library(VennDiagram)
# 
# venn_list <- list(
#   Set1 = rownames(DEG0),
#   Set2 = rownames(DEG1)
# )
# 
# venn.plot <- venn.diagram(
#   x = venn_list,
#   category.names = c("List 1", "List 2"),
#   filename = NULL  # Use NULL if you don't want to save the plot as a file
# )
# grid.draw(venn.plot)
# 

