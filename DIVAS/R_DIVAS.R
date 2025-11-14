library(R.matlab)
library(DIVAS)

data_path <- system.file("extdata", "toyDataThreeWay.mat", package = "DIVAS") 
data <- R.matlab::readMat(data_path) 
str(data$datablock)

datablock_list <- list(
 Block1 = data$datablock[1,1][[1]][[1]],
 Block2 = data$datablock[1,2][[1]][[1]],
 Block3 = data$datablock[1,3][[1]][[1]]
)

print(paste("Block 1 dimensions:", dim(datablock_list[[1]])))
print(paste("Block 2 dimensions:", dim(datablock_list[[2]])))
print(paste("Block 3 dimensions:", dim(datablock_list[[3]])))

divas_results <- DIVAS::DIVASmain(datablock_list)

