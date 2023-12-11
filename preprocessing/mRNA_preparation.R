setwd('/home/che82/athan/MMBiomarker/data/')

file_sample <- read.csv('./TCGA-PanCancerData-raw/sample_sheet_mrna.csv', header=T)

count_file <- list.files('./TCGA-PanCancerData-raw/mRNA/', pattern='*.tsv', recursive=T)
count_file_name <- strsplit(count_file, split='/')
count_file_name <- sapply(count_file_name, function(x){x[2]})

# get raw counts
fileids <- c()
for (i in 1:length(count_file)){
  path = paste0('./TCGA-PanCancerData-raw/mRNA/', count_file[i])
  data <- read.delim(path, fill=T, header=FALSE, row.names=1)
  colnames(data) <- data[2,] #
  data <- data[-c(1:6), ] # remove rows that are not gene expressions
  data <- data[data['gene_type']=='protein_coding',] # Remove non-protein coding genes (retain mRNA)
  data <- data[!duplicated(data$gene_name),] # remove duplicated rows
  rownames(data) <- data$gene_name # change row names to gene names
  rownames(data) = gsub('[.]', '-', rownames(data))
  data <- data['unstranded'] # Choose which data column to use. here we choose raw counts

  # NOTE some samples have multiple files, right now I just choose whichever file comes first for those samples.
  if (i==1) {
    nrows = dim(data)[1]
    mtx = data.frame(matrix(nrow=nrows, ncol=0))
  }
  sample_id <- file_sample$Sample.ID[which(file_sample$File.Name==count_file_name[i])] # set col name as sample id
  if (sample_id %in% colnames(mtx)) {
    next
  }
  colnames(data) <- sample_id
  mtx <- cbind(mtx, data)
  fileids <- c(fileids, file_sample$File.ID[which(file_sample$File.Name==count_file_name[i])])
}

# save
write.csv(mtx, './TCGA-PanCancerData-intermediate/mRNA.csv')
write.csv(fileids, './TCGA-PanCancerData-intermediate/fileids_mRNA.csv')
