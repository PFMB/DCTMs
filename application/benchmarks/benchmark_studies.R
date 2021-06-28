######################### Benchmark studies ##############################

library(parallel)
rm(list=ls())
max_epoch = 3000
nrCores = 20
nrsims = 20

datasets_to_run <- c("airfoil",
                     "boston",
                     "forestfire",
                     "diabetes")

for(M in c(1,4,10,25)){
  
  ################# Forest Fire #################
  
  if("forestfire" %in% datasets_to_run){
    
    ff <- read.csv("data/forestfires/forestfires.csv")
    # transform outcome as described in 
    # the repository (https://archive.ics.uci.edu/ml/datasets/forest+fires)
    ff$area <- log(ff$area + 1)
    
    set.seed(42)
    
    index_train <- sample(1:nrow(ff), round(nrow(ff)*0.75))
    
    train <- ff[index_train,]
    test <- ff[setdiff(1:nrow(ff), index_train),]
    y_train <- train$area
    y_test <- test$area
    
    train <- model.matrix(~ 0 + ., data = train)
    test <- model.matrix(~ 0 + ., data = test)
    
    Vs <- setdiff(colnames(train),"area")
    
    form_h2 <- paste0("~ 1", 
                      " + deep_mu(",
                      paste(Vs, collapse=", "), ")")
    
    form_h1 <- paste0("~ 1", 
                      " + deep_mu(",
                      paste(Vs, collapse=", "), ")")
    
    res <- mclapply(1:nrsims, function(sim_iteration){
      
      library(devtools)
      load_all("~/deepregression_master/deepregression/R")
      
      deep_mod <- function(x) x %>% 
        layer_dense(units = 16, activation = "tanh", use_bias = FALSE) %>%
        layer_dense(units = 4, activation = "tanh") %>%
        layer_dense(units = 1, activation = "linear")
      
      mod_deep <- deepregression(y = y_train, 
                                 list_of_formulae = list(as.formula(form_h2),
                                                         as.formula(form_h1)),
                                 list_of_deep_models = list(deep_mu = deep_mod),
                                 data = as.data.frame(train),
                                 family = "transformation_model",
                                 cv_folds = 5,
                                 optimizer = optimizer_rmsprop(),
                                 order_bsp = M)
      
      st <- Sys.time()
      
      cvres <- mod_deep %>% cv(epochs = max_epoch)
      
      (ep <- stop_iter_cv_result(cvres))
      
      mod_deep %>% fit(epochs = ep, 
                       verbose = FALSE, 
                       view_metrics = FALSE,
                       validation_split = NULL)
      
      et <- Sys.time()
      
      (ll <- log_score(mod_deep, as.data.frame(test), y_test, summary_fun = mean))
      
      c (ll, as.numeric(difftime(et,st,units="mins")))
      
    }, mc.cores = nrCores)
    
    ress <- try(data.frame(do.call("rbind",res)))
    
    name = paste0(M, "_ff.RDS")
    
    if(class(ress)=="try-error"){
      saveRDS(res, file=name)
    }else{
      ress[,1] <- -ress[,1]
      colnames(ress) <- c("NLL", "time")
      saveRDS(ress, file=name)
      cat(mean(ress$NLL),",",stats::sd(ress$NLL))
    }
    
  }
  
  ################# Diabetes #################
  
  if("diabetes" %in% datasets_to_run){
    
    # load data saved in python with specific random_state
    
    y_train <- read.csv("data/diabetes/y_train_diabetes.csv", header=F)
    y_test <- read.csv("data/diabetes/y_test_diabetes.csv", header=F)
    x_train <- read.csv("data/diabetes/x_train_diabetes.csv", header=F)
    x_test <- read.csv("data/diabetes/x_test_diabetes.csv", header=F)
    
    set.seed(42)
    
    # define measures
    
    Vs <- paste0("V",1:10)
    form_h2 <- paste0("~ 1 + s(V3)", 
                      "+ ",
                      " d(",
                      paste(Vs, collapse=", "), ")")
    
    form_h1 <- paste0("~ 1 + ", "d(",
                      paste(Vs, collapse=", "), ")")
    
    deep_mod <- function(x) x %>% 
      layer_dense(units = 4, activation = "tanh") %>% 
      layer_dense(units = 1, activation = "linear")
    
    res <- mclapply(1:nrsims, function(sim_iteration){
      
      library(devtools)
      load_all("~/deepregression_master/deepregression/R")
      
      mod_deep <- deepregression(y = y_train$V1, 
                                 list_of_formulae = list(as.formula(form_h2),
                                                         as.formula(form_h1)),
                                 list_of_deep_models = list(deep_mod, 
                                                            deep_mod),
                                 data = x_train,
                                 family = "transformation_model",
                                 cv_folds = 5,
                                 optimizer = optimizer_adadelta(lr = 0.1),
                                 order_bsp = M)
      
      st <- Sys.time()
      
      cvres <- mod_deep %>% cv(epochs = max_epoch)
      
      (ep <- stop_iter_cv_result(cvres))
      
      mod_deep %>% fit(epochs = ep, 
                       verbose = FALSE, view_metrics = FALSE,
                       validation_split = NULL)
      
      et <- Sys.time()
      
      (ll <- log_score(mod_deep, as.data.frame(x_test), y_test$V1, summary_fun = mean))
      
      c(ll, as.numeric(difftime(et,st,units="mins")))
      
    }, mc.cores = nrCores)
    
    ress <- try(data.frame(do.call("rbind",res)))
    
    name=paste0(M, "_diabetes.RDS")
    
    if(class(ress)=="try-error"){
      saveRDS(res, file=name)
    }else{
      ress[,1] <- -ress[,1]
      colnames(ress) <- c("NLL", "time")
      saveRDS(ress, file=name)
      cat(mean(ress$NLL),",",stats::sd(ress$NLL))
    }
    
  }
  
  ################# Boston #################
  
  if("boston" %in% datasets_to_run){
    
    # load data saved in python with specific random_state
    
    y_train <- read.csv("data/boston/y_train.csv", header=F)
    y_test <- read.csv("data/boston/y_test.csv", header=F)
    x_train <- read.csv("data/boston/x_train.csv", header=F)
    x_test <- read.csv("data/boston/x_test.csv", header=F)
    
    set.seed(42)
    
    # define measures
    
    Vs <- paste0("V",1:13)
    form_h2 <- paste0("~ 1 ", 
                      "+",
                      paste(Vs, collapse=" + "),
                      " + s(",
                      paste(Vs[c(-4,-9)], collapse=") + s("), ")",
                      "+ deep_mu(",
                      paste(Vs, collapse=", "), ")")
    
    form_h1 <- paste0("~ 1", 
                      "+",
                      "deep_sig(",
                      paste(Vs, collapse=", "), ")"
    )
    
    
    res <- mclapply(1:nrsims, function(sim_iteration){
      
      library(devtools)
      load_all("~/deepregression_master/deepregression/R")
      
      deep_mod <- function(x) x %>% 
        layer_dense(units = 32, activation = "tanh", use_bias = FALSE) %>%
        layer_dense(units = 16, activation = "tanh") %>% 
        layer_dense(units = 4, activation = "tanh") %>% 
        layer_dense(units = 1, activation = "linear")
      
      deep_mod2 <- function(x) x %>% 
        layer_dense(units = 2, activation = "tanh", use_bias = FALSE) %>%
        layer_dense(units = 1, activation = "linear")
      
      mod_deep <- deepregression(y = y_train$V1, 
                                 list_of_formulae = list(as.formula(form_h2),
                                                         as.formula(form_h1)),
                                 list_of_deep_models = list(deep_mu=deep_mod, 
                                                            deep_sig=deep_mod2),
                                 data = x_train,
                                 family = "transformation_model",
                                 cv_folds = 5,
                                 optimizer = optimizer_adadelta(lr = 0.1),
                                 order_bsp = M)
      
      st <- Sys.time()
      
      cvres <- mod_deep %>% cv(epochs = max_epoch)
      
      (ep <- stop_iter_cv_result(cvres))
      
      mod_deep %>% fit(epochs = ep, 
                       verbose = FALSE, view_metrics = FALSE,
                       validation_split = NULL)
      
      et <- Sys.time()
      
      (ll <- log_score(mod_deep, as.data.frame(x_test), y_test$V1, summary_fun = mean))
      
      c(ll, as.numeric(difftime(et,st,units="mins")))
      
    }, mc.cores = nrCores)
    
    ress <- try(data.frame(do.call("rbind",res)))
    
    name=paste0(M, "_boston.RDS")
    
    if(class(ress)=="try-error"){
      saveRDS(res, file=name)
    }else{
      ress[,1] <- -ress[,1]
      colnames(ress) <- c("NLL", "time")
      saveRDS(ress, file=name)
      cat(mean(ress$NLL),",",stats::sd(ress$NLL))
    }
    
  }
  
  ################# Airfoil #################
  
  if("airfoil" %in% datasets_to_run){
    
    airfoil <- read.table("data/airfoil/airfoil_self_noise.dat")
    
    set.seed(42)
    
    index_train <- sample(1:nrow(airfoil), round(nrow(airfoil)*0.75))
    
    train <- airfoil[index_train,]
    test <- airfoil[setdiff(1:nrow(airfoil), index_train),]
    
    # define measures
    
    Vs <- paste0("V",1:5)
    form_h2 <- paste0("~ 1",
                      " + s(",
                      paste(Vs[c(-3,-4)], collapse=") + s("), ") + d(",
                      paste(Vs, collapse=", "), ")")
    
    form_h1 <- paste0("~ 1 + ", "d(",
                      paste(Vs, collapse=", "), ")")
    
    
    
    res <- mclapply(1:nrsims, function(sim_iteration){
      
      library(devtools)
      load_all("~/deepregression_master/deepregression/R")
      
      deep_mod <- function(x) x %>% 
        layer_dense(units = 16, activation = "tanh", use_bias = FALSE) %>%
        layer_dense(units = 4, activation = "tanh") %>% 
        layer_dense(units = 1, activation = "linear")
      
      mod_deep <- deepregression(y = train$V6, 
                                 list_of_formulae = list(as.formula(form_h2),
                                                         as.formula(form_h1)),
                                 list_of_deep_models = list(deep_mod, 
                                                            deep_mod),
                                 data = train[,1:5],
                                 family = "transformation_model",
                                 cv_folds = 5,
                                 optimizer = optimizer_adadelta(lr = 0.1),
                                 order_bsp = M)
      
      st <- Sys.time()
      
      cvres <- mod_deep %>% cv(epochs = max_epoch)
      
      (ep <- stop_iter_cv_result(cvres))
      
      mod_deep %>% fit(epochs = ep, 
                       verbose = FALSE, view_metrics = FALSE,
                       validation_split = NULL)
      
      et <- Sys.time()
      
      (ll <- log_score(mod_deep, as.data.frame(test), test[,6], summary_fun = mean))
      
      c(ll, as.numeric(difftime(et,st,units="mins")))
      
    }, mc.cores = nrCores)
    
    ress <- try(data.frame(do.call("rbind",res)))
    
    name=paste0(M, "_airfoil.RDS")
    
    if(class(ress)=="try-error"){
      saveRDS(res, file=name)
    }else{
      ress[,1] <- -ress[,1]
      colnames(ress) <- c("NLL", "time")
      saveRDS(ress, file=name)
      cat(mean(ress$NLL),",",stats::sd(ress$NLL))
    }
    
  }
  
  ###########################################
  
}

# library(dplyr)
# library(tidyr)
# library(xtable)
# 
do.call("rbind", lapply(list.files(pattern = ".RDS"), function(x){
  res <- readRDS(x)
  M <- gsub("(.*)_(.*)\\.RDS","\\1",x)
  dataset <- gsub("(.*)_(.*)\\.RDS","\\2",x)
  return(cbind(data.frame(M=M, dataset=dataset), res))
  #paste0(signif(mean(res$NLL),3)," (",signif(stats::sd(res$NLL),3),")")
  })) %>% group_by(M, dataset) %>% summarise(res = paste0(signif(mean(NLL),3),
                                                          " (",signif(stats::sd(NLL),3),
                                                          ")")) %>%
  spread(key = M, res, fill = NA, convert = FALSE) %>%
  relocate(dataset, `4`, `10`, `25`) %>%
  xtable()
