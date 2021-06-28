rm(list = ls())
set.seed(1)
library(data.table)
library(stringr)
library(parallel)
library(mlt)
library(tbm)
source("tbm_score_fun.R")

# Download "Aligned & Cropped Faces" (107 MB) from https://susanqq.github.io/UTKFace/
d <- readRDS("d_cleaned.RDS")

###
cv_d_set <- readRDS("utk_faces_cv_id.RDS")
nr_cluster <- 4

# cv_d_set <- lapply(1:nr_cluster, function(x, n = nrow(d)) {
#   id <- 1:n
#   set.seed(x)
#   train_id <- sample(id, n * 0.7)
#   test_id <- id[!id %in% train_id]
#   res <- list("train_id" = train_id, "test_id" = test_id)
#   attr(res, "seed") <- x
#   attr(res, "info") <- Sys.info()
#   attr(res, "time") <- Sys.time()
#   res
# })


latent_images <- readRDS("latent_feature_images.RDS")
latent_images <- as.data.frame(latent_images)
pca_li <- princomp(latent_images)
#plot(cumsum(pca_li$sdev^2)/sum(pca_li$sdev^2))

# choose as many features to explain 90% of the variance
latent_images <- pca_li$scores[,1:50]
colnames(latent_images) <- paste0("li_", 1:ncol(latent_images))

res_TBM <- lapply(cv_d_set, function(d_id){
  
  d_train <- d[d_id$train_id,]
  d_test <- d[d_id$test_id,]
  lat_feat_train <- latent_images[d_id$train_id,]
  lat_feat_test <- latent_images[d_id$test_id,]
  
  res <- stm_ctm_boost_eval(train_x = cbind(d_train[,c("gender", "race", "days_to_fin")],
                                            lat_feat_train), 
                            train_y = d_train$age, 
                            test_x = cbind(d_test[,c("gender", "race", "days_to_fin")],
                                           lat_feat_test), 
                            test_y = d_test$age, outcome_name = "age", 
                            lin_names = c(c("gender", "race", "days_to_fin"), colnames(lat_feat_train)),
                            scale_feature_names = c("days_to_fin", colnames(lat_feat_train)))
  
  return(res)
  
})

saveRDS(res_TBM, file="res_TBM.RDS")

# readRDS("res_TBM.RDS")
pf <- function(x) paste0(signif(mean(x),3)," (",signif(sd(x),3),")")
pf(as.numeric(sapply(readRDS("res_TBM.RDS"), "[[", 1)))
pf(as.numeric(sapply(readRDS("res_TBM.RDS"), "[[", 2)))
