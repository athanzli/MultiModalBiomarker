# install.packages('rjson')
# install.packages('tidyverse')

library('rjson')
library('tidyverse')

setwd('C:/Users/athan/OneDrive/Desktop/MMBiomarker/MMBiomarker/data/TCGA-LIHC_from_GDC_raw')

json <- jsonlite::fromJSON('./metadata.cart.2023-10-30.json')
sample_id <- sapply(json$associated_entities, function(x){x[,1]})
file_sample <- data.frame(sample_id, file_name=json$file_name)

count_file <- list.files('gdc_download_20231030_215821.291656', pattern='*.tsv', recursive=TRUE)
count_file_name <- strsplit(count_file, split='/')
count_file_name <- sapply(count_file_name, function(x){x[2]})

# get raw counts
for (i in 1:length(count_file)){
  path = paste0('gdc_download_20231030_215821.291656//', count_file[i])
  data <- read.delim(path, fill=TRUE, header=FALSE, row.names=1)
  colnames(data) <- data[2,] # 
  data <- data[-c(1:6), ] # remove rows that are not gene expressions
  # Remove non-protein coding genes (retain mRNA)
  data <- data[data['gene_type']=='protein_coding',]
  # remove duplicated rows
  data <- data[!duplicated(data$gene_name),]
  # change row names to gene names
  rownames(data) <- data$gene_name
  rownames(data) = gsub('[.]', '-', rownames(data))
  # Choose which data column to use 
  data <- data['unstranded'] # here we choose raw counts
  # set col name as sample id
  colnames(data) <- file_sample$sample_id[which(file_sample$file_name==count_file_name[i])]
  if (i==1) {
    nrows = dim(data)[1]
    mtx = data.frame(matrix(nrow=nrows, ncol=0))
  }
  mtx <- cbind(mtx, data)
}

# save
write.csv(mtx, 'TCGA_LIHC_mRNA_raw_counts.csv')

