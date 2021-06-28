rm(list = ls())
set.seed(1)
library(data.table)
library(stringr)
library(parallel)

layer_info <- function(deepreg, idx_2 = 0) {
  ## help function
  # get number of weights and layer name from deepreg object, first ones are the deepest layers
  L <- as.integer(length(deepreg$model$layers) - 1)
  cat("No. of layers:", L + 1,"\n")# zero indexing
  sapply(0L:L, function(idx) {
    wght <- deepreg$model$get_layer(index = idx)$get_weights()
    wght <- ifelse(length(wght) == 0, 0, sapply(wght, length))
    nm <- paste0(deepreg$model$layers[[
      ifelse(idx + 1L < L - idx_2,idx + 1L,L - 1 - idx_2)]]$get_config()[["name"]],
      "_layer_" ,idx)
    setNames(wght, nm)
  })
}

# Download "Aligned & Cropped Faces" (107 MB) from https://susanqq.github.io/UTKFace/
p <- "~/deepregression_master/deepregression/R"
images <- list.files("UTKFace")

# 23708 200x200 RGB faces 

# The labels of each face image is embedded in the file name, 
# formated like [age]_[gender]_[race]_[date&time].jpg
# 
# [age] is an integer from 0 to 116, indicating the age
# [gender] is either 0 (male) or 1 (female)
# [race] is an integer from 0 to 4, denoting White, Black, Asian, 
# Indian, and Others (like Hispanic, Latino, Middle Eastern).
# [date&time] is in the format of yyyymmddHHMMSSFFF, 
# showing the date and time an image was collected to UTKFace


d <- data.table(str_split(images, "_", simplify = TRUE))
setnames(d, old = c("V1","V2","V3","V4"), new = c("age","gender","race","date_time"))
d[, age := as.numeric(age)]
d[, gender := as.factor(gender)]
d[, race := as.factor(race)]
d[, image := paste0(path,images)]
d[, date := as.Date(substr(date_time, 1, nchar("yyyymmdd")), "%Y%m%d")]

# too many meaningless levels, maybe coarsen into intervals?
#d[, time := as.factor(substr(date_time, nchar("yyyymmdd") + 1, nchar("yyyymmddHHMMSSFFF")))]

## seem to have missing "race" entries and thus are excluded ##
ms <- c("61_1_20170109142408075.jpg.chip.jpg","61_1_20170109150557335.jpg.chip.jpg", 
        "39_1_20170116174525125.jpg.chip.jpg")
d <- d[!image %in% paste0(path,ms),]
d[, race := droplevels(race)] # 23705 faces

d[,days_to_fin := as.numeric(max(date) - date)]
#d <- d[sample(1:nrow(d) , 1e3),] # to reduce runtime

# saveRDS(d, file="d_cleaned.RDS")

###

nr_cluster <- 4

cv_d_set <- lapply(1:nr_cluster, function(x, n = nrow(d)) {
  id <- 1:n
  set.seed(x)
  train_id <- sample(id, n * 0.7)
  test_id <- id[!id %in% train_id]
  res <- list("train_id" = train_id, "test_id" = test_id)
  attr(res, "seed") <- x
  attr(res, "info") <- Sys.info()
  attr(res, "time") <- Sys.time()
  res
})



#save(cv_d_set, file = paste0(path_1,"utk_faces_cv_id.RData"))

###

dctm_faces <- function(d_id) {
  
  ################
  
  cnn_block <- function(filters, kernel_size, pool_size, rate, input_shape = NULL){
    function(x){
      x %>% 
        layer_conv_2d(filters, kernel_size, padding="same", input_shape = input_shape) %>% 
        layer_activation(activation = "relu") %>% 
        layer_batch_normalization() %>% 
        layer_max_pooling_2d(pool_size = pool_size) %>% 
        layer_dropout(rate = rate)
    }
  }
  
  ################
  
  d_train <- d[d_id$train_id,]
  d_test <- d[d_id$test_id,]
  
  devtools::load_all(p)
  
  bsp_M <- 25
  print_2_screen <- FALSE
  se <- .Random.seed
  
  mf_h_1 <- as.formula("~ race + gender") # days_to_fin only in h_2 for interpretation purposes
  mf_h_2 <- as.formula("~ s(days_to_fin, bs = 'bs', k = 7) + race + gender")
  
  ###  h_1 = struc_only & h_2 = struc_only ###
  
  dctm <- deepregression(y = d_train$age, 
                         family = "transformation_model",
                         list_of_formulae = list(mf_h_2, mf_h_1),
                         data = d_train,
                         list_of_deep_models = NULL,
                         optimizer = optimizer_adam(lr = 0.01), 
                         tf_seed = 1,
                         order_bsp = bsp_M,
                         base_distribution = tfd_logistic(loc = 0, scale = 1),
                         #addconst_interaction = 0,
                         sp_scale = nrow(d_train)
                         #df = 9
  )
  
  dctm %>% fit(
    epochs = 1e4, validation_split = 0.2,
    verbose = print_2_screen, batch_size = 64,
    callbacks = list(
      callback_early_stopping(patience = 300),
      callback_reduce_lr_on_plateau(patience = 200)
    ),
    view_metrics = FALSE
  )
  
  dctm1 <- list("pls" = log_score(dctm, data = d_test, this_y = d_test$age, summary_fun = mean),
                "shift" = get_shift(dctm),
                "plot" = plot(dctm, plot = FALSE),
                "theta" = get_theta(dctm),
                "metrics" = keras:::to_keras_training_history(dctm$model$history)$metrics)
  
  # get basic info about the network structure
  #layer_info(dctm_1)
  
  ## interaction effects (h_1) := (intercept + nlevels(d$race) - 1 + nlevels(d$gender) - 1) * (M + 1)
  w_h1 <- dctm$model$get_layer(index = 15L)$get_weights()
  
  ## shift effects (h_2) := (intercept + k = 7 + nlevels(d$race) - 1 + nlevels(d$gender) - 1) = 12
  w_h2 <- dctm$model$get_layer(index = 14L)$get_weights()
  
  # number of learned paramters (192 + 12)
  length(unlist(get_weights(dctm$model)))
  
  ### h_1 = struc + image ||| h_2 = struc + image ###
  
  # DNN for images (1 conv öayer)
  
  # nn_big <- function(x) x %>% 
  #   # Conv-Block 1
  #   layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
  #                 input_shape = shape(200, 200, 3), 
  #                 kernel_regularizer = regularizer_l2(l = 0.0001)) %>% 
  #   layer_batch_normalization() %>%
  #   layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #   layer_dropout(rate = 0.2) %>%
  #   # FC layer
  #   layer_flatten() %>%
  #   layer_dense(units = 32, activation = "relu") %>%
  #   layer_dropout(rate = 0.2) %>%
  #   layer_dense(units = 1)
  
  cnn1 <- cnn_block(filters = 16, kernel_size = c(3,3), pool_size = c(3,3), rate = 0.25,
                    shape(200, 200, 3))
  cnn2 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  cnn3 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  
  nn_big <- function(x){
    x %>% cnn1() %>% cnn2() %>% cnn3() %>% 
      # branch end
      layer_flatten() %>% 
      layer_dense(128) %>% 
      layer_activation(activation = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = 0.5) %>% 
      layer_dense(2)
  }
  
  tt_list <- list( ~ + nn(image))[rep(1,2)]
  
  dctm <- deepregression(y = d_train$age, 
                         family = "transformation_model",
                         list_of_formulae = list(mf_h_2, mf_h_1),
                         data = d_train,
                         tf_seed = 1,
                         order_bsp = bsp_M,
                         base_distribution = tfd_logistic(loc = 0, scale = 1),
                         image_var = list(image = list(c(200,200,3))),
                         list_of_deep_models = list(nn = nn_big),
                         train_together = tt_list,
                         split_between_shift_and_theta = c(1L,1L),
                         optimizer = optimizer_adam(), 
                         #addconst_interaction = 0,
                         sp_scale = nrow(d_train)
                         #df = 9
  )
  
  # get basic info about the network structure
  #layer_info(dctm)
  
  # take weights from structured only model such that dctm start optimization from warmed up point
  # dctm$model$get_layer(index = 193L)$set_weights(w_h2)
  # dctm$model$get_layer(index = 197L)$set_weights( # fill nn pred thetas with random numbers
  #   list(rbind(w_h1[[1]],matrix(rnorm(1 * (bsp_M + 1)),ncol = 1)))) 
  
  # 1-175 Resnet, do not freeze last dense layer (176)
  #freeze_weights(dctm$model, from = 1, to = 175)
  
  dctm %>% fit(
    epochs = 4e2, validation_split = 0.2,
    verbose = print_2_screen, batch_size = 25,
    callbacks =   list(
      callback_learning_rate_scheduler(
        tf$keras$experimental$CosineDecayRestarts(.02, 30, t_mul = 2, m_mul = .7)
      ),                  
      callback_early_stopping(patience = 30)
    ),
    view_metrics = FALSE
  )
  
  dctm2 <- list("pls" = log_score(dctm, data = d_test, this_y = d_test$age, summary_fun = mean),
                "shift" = get_shift(dctm),
                "plot" = plot(dctm, plot = FALSE),
                "theta" = get_theta(dctm),
                "metrics" = keras:::to_keras_training_history(dctm$model$history)$metrics)
  
  ### h_1 = struc + image || h_2 = struc ###
  
  mf_h_1 <- as.formula("~ race + gender + nn(image)")
  
  # nn_big <- function(x){
  #   application_resnet50(
  #     input_tensor = x,
  #     weights = 'imagenet',
  #     include_top = FALSE)$output %>%
  #     layer_global_average_pooling_2d() %>% 
  #     layer_dense(units = 1) # one for h_1 only
  # }
  

  cnn1 <- cnn_block(filters = 16, kernel_size = c(3,3), pool_size = c(3,3), rate = 0.25,
                    shape(200, 200, 3))
  cnn2 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  cnn3 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  
  nn_big <- function(x){
    x %>% cnn1() %>% cnn2() %>% cnn3() %>% 
      # branch end
      layer_flatten() %>% 
      layer_dense(128) %>% 
      layer_activation(activation = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = 0.5) %>% 
      layer_dense(1)
  }
  
  dctm <- deepregression(y = d_train$age, 
                         family = "transformation_model",
                         list_of_formulae = list(mf_h_2, mf_h_1),
                         data = d_train,
                         tf_seed = 1,
                         order_bsp = bsp_M,
                         base_distribution = tfd_logistic(loc = 0, scale = 1),
                         image_var = list(image = list(c(200,200,3))),
                         list_of_deep_models = list(nn = nn_big),
                         optimizer = optimizer_adam(), 
                         #addconst_interaction = 0,
                         sp_scale = nrow(d_train)
                         #df = 9
  )
  
  # get basic info about the network structure
  #layer_info(dctm)
  # 
  # dctm$model$get_layer(index = 192L)$set_weights(w_h2)
  # dctm$model$get_layer(index = 193L)$set_weights(list(rbind(w_h1[[1]],
  #                                                           matrix(rnorm(1 * (bsp_M + 1)),ncol = 1))))
  
  #freeze_weights(dctm$model, from = 1, to = 172)
  
  dctm %>% fit(
    epochs = 4e2, validation_split = 0.2,
    verbose = print_2_screen, batch_size = 25,# steps_per_epoch = 10,
    callbacks =   list(
      callback_learning_rate_scheduler(
        tf$keras$experimental$CosineDecayRestarts(.02, 30, t_mul = 2, m_mul = .7)
      ),                  
      callback_early_stopping(patience = 30)
    ),
    view_metrics = FALSE
  )
  
  dctm3 <- list("pls" = log_score(dctm, data = d_test, this_y = d_test$age, summary_fun = mean),
                "shift" = get_shift(dctm),
                "plot" = plot(dctm, plot = FALSE),
                "theta" = get_theta(dctm),
                "metrics" = keras:::to_keras_training_history(dctm$model$history)$metrics)
  
  ### h_1 = struc || h_2 = struc + image ###
  
  mf_h_1 <- as.formula("~ race + gender")
  mf_h_2 <- as.formula("~ s(days_to_fin, bs = 'bs', k = 7) + race + gender + nn(image)")
  
  cnn1 <- cnn_block(filters = 16, kernel_size = c(3,3), pool_size = c(3,3), rate = 0.25,
                    shape(200, 200, 3))
  cnn2 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  cnn3 <- cnn_block(filters = 32, kernel_size = c(3,3), pool_size = c(2,2), rate = 0.25)
  
  nn_big <- function(x){
    x %>% cnn1() %>% cnn2() %>% cnn3() %>% 
      # branch end
      layer_flatten() %>% 
      layer_dense(128) %>% 
      layer_activation(activation = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = 0.5) %>% 
      layer_dense(1)
  }
  
  dctm <- deepregression(y = d_train$age, 
                         family = "transformation_model",
                         list_of_formulae = list(mf_h_2, mf_h_1),
                         data = d_train,
                         tf_seed = 1,
                         order_bsp = bsp_M,
                         base_distribution = tfd_logistic(loc = 0, scale = 1),
                         image_var = list(image = list(c(200,200,3))),
                         list_of_deep_models = list(nn = nn_big),
                         optimizer = optimizer_adam(), 
                         #addconst_interaction = 0,
                         sp_scale = nrow(d_train)
                         #df = 9
  )
  
  # get basic info about the network structure
  #layer_info(dctm)
  
  # dctm$model$get_layer(index = 190L)$set_weights(w_h2)
  # dctm$model$get_layer(index = 194L)$set_weights(w_h1)
  
  #freeze_weights(dctm$model, from = 1, to = 172)
  
  dctm %>% fit(
    epochs = 4e2, validation_split = 0.2,
    verbose = print_2_screen, batch_size = 25,# steps_per_epoch = 10,
    callbacks =   list(
      callback_learning_rate_scheduler(
        tf$keras$experimental$CosineDecayRestarts(.02, 30, t_mul = 2, m_mul = .7)
      ),                  
      callback_early_stopping(patience = 30)
    ),
    view_metrics = FALSE
  )
  
  dctm4 <- list("pls" = log_score(dctm, data = d_test, this_y = d_test$age, summary_fun = mean),
                "shift" = get_shift(dctm),
                "plot" = plot(dctm, plot = FALSE),
                "theta" = get_theta(dctm),
                "metrics" = keras:::to_keras_training_history(dctm$model$history)$metrics)
  
  res <- list("dctm1" = dctm1, "dctm2" = dctm2, "dctm3" = dctm3, "dctm4" = dctm4)
  attr(res, "seed") <- se
  attr(res, "info") <- Sys.info()
  attr(res, "time") <- Sys.time()
  return(res)
  
}

# cl <- makeForkCluster(nr_cluster) # load_all() does not work with makeCluster() locally
# clusterExport(cl, list("d","p"))
# res <- parLapply(cl, cv_d_set, dctm_faces)
# stopCluster(cl)



res <- mclapply(cv_d_set, dctm_faces, mc.cores = 2)
#stopCluster(cl)
saveRDS(res, file = "utk_faces_dctm_res.RDS")
res_pls <- sapply(1:4, function(idx) {
  dctm <- lapply(res, `[[`, paste0("dctm",idx))
  dctm_pls <- sapply(dctm, `[[`, "pls")
  c("mean_pls" = mean(dctm_pls), "sd_pls" = stats::sd(dctm_pls))
})
print(res_pls)

##################
# create latent features for TBM
library(keras)
model <- application_vgg16(weights = 'imagenet', include_top = FALSE)
create_feat <- function(x){
  img <- image_load(x, target_size = c(33,33))
  x <- image_to_array(img)
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)
  return((model %>% predict(x))[1,1,1,])
}
latent_images <- do.call("rbind", lapply(1:nrow(d), function(i) create_feat(d$image[i])))
saveRDS(latent_images, file="latent_feature_images.RDS")
##################
