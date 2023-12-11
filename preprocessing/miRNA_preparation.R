# install.packages('rjson')
# install.packages('tidyverse')

setwd('/home/che82/athan/MMBiomarker_data/TCGA-PanCancerData-raw')

file_sample <- read.csv('sample_sheet_mirna.csv', header=T)

count_file <- list.files('./miRNA/', pattern='*mirbase21.mirnas.quantification.txt', recursive=TRUE)
count_file_name <- strsplit(count_file, split='/')
count_file_name <- sapply(count_file_name, function(x){x[2]})

# get miRNA expression
fileids <- c()
for (i in 1:length(count_file)){
  path = paste0('./miRNA/', count_file[i])
  data <- read.delim(path, fill=T, header=T, row.names=1)
  data <- data['reads_per_million_miRNA_mapped'] # Choose which data column to use 
  # set col name as sample id
  colnames(data) <- file_sample$Sample.ID[which(file_sample$File.Name==count_file_name[i])]
  if (i==1) {
    nrows = dim(data)[1]
    mtx = data.frame(matrix(nrow=nrows, ncol=0))
  }
  if (colnames(data) %in% colnames(mtx)) {
    print(i)
    next
  }
  mtx <- cbind(mtx, data)
  fileids <- c(fileids, file_sample$File.ID[which(file_sample$File.Name==count_file_name[i])])
}

# save
write.csv(mtx, '../TCGA-PanCancerData-intermediate/miRNA.csv')
write.csv(fileids, '../TCGA-PanCancerData-intermediate/fileids_miRNA.csv')
