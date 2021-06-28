#library(tidyverse)
#library(jsonlite)
#library(lubridate)
#library(tidytext)
#library(tm)
#theme_set(theme_bw())
#library(ggplot2)
#library(devtools)
library(data.table)
library(tbm)
library(parallel)

###### 
### Transformation Boosting Machines application to the movies data set
######

rm(list = ls())
set.seed(1)
data_list <- readRDS("./application/data_splitted.RDS")

###### 
### Preprocessing (as conducted in movies.R)
######
# 
# # read in data
# movies <- read_csv("./application/movies.csv")
# 
# # get genres
# movies <- movies %>%
#   filter(original_language == "en" & status == "Released") %>% 
#   # filter(nchar(genres)>2) %>%
#   mutate(genres = lapply(genres, function(x) fromJSON(x)$name)) %>%  
#   select(genres, budget, overview, popularity, 
#          production_countries, release_date, 
#          revenue, runtime, vote_average, 
#          vote_count) %>% 
#   filter(vote_count > 0) %>% 
#   mutate(release_date = as.numeric(as.Date("2020-01-01") - as.Date(release_date)))
# 
# genres <- movies %>%  unnest(genres, .name_repair = "unique") %>% select(genres) 
# 
# movies <- movies[unlist(sapply(movies$genres, function(x) 
#   length(x) > 1 | (!("TV Movie" %in% x) & !("Foreign" %in% x)))),]
# 
# all_genres <- sort(unique(unlist(movies$genres)))
# genres_wide <- t(sapply(movies$genres,
#                         function(x) 
#                           colSums(model.matrix(~ -1 + genre, 
#                                                data = data.frame(genre = factor(x, levels = all_genres))))
# ))
# 
# colnames(genres_wide) <- gsub(" ", "_", colnames(genres_wide))
# 
# movies <- cbind(movies %>% select(-genres), genres_wide)        
# 
# d <- data.table(movies)
# 
# d[, overview := NULL] # model-based boosting cant handle unstructed (i.e. pure text data)
# d[, production_countries := NULL] # excluded here since it was excluded when estimated with deepregression()

# if one wants to include production countries, one can un-comment the following lines
# to get a dummy coding for this variable (most countries have more than one production country)

# count <- lapply(movies$production_countries, function(x) {
#   y <- unlist(strsplit(x,split = "[{}]"))
#   sort(y[grepl("name", y)])
# })
# all_count <- sort(unique(unlist(count)))
# count_wide <- t(sapply(count,
#                         function(x) 
#                           colSums(model.matrix(~ -1 + prod_count, 
#                                                data = data.frame(prod_count = factor(x, levels = all_count))))
# ))
# colnames(count_wide) <- paste0("prod_",sub('.*name\": \"', '', colnames(count_wide)))
# colnames(count_wide) <- substr(colnames(count_wide), 1, nchar(colnames(count_wide)) - 1)
# colnames(count_wide)  <- gsub("[[:blank:]]", "", colnames(count_wide) )
# 
# movies <- cbind(movies %>% select(-production_countries), count_wide)   

######

###### 
### STM/CTM Boosting with 10-repititions
######

# repeat analysis 10 times
res_boost <- mclapply(1:length(data_list), function(repl) {
  
  train <- as.data.table(data_list[[repl]]$train)
  test <- as.data.table(data_list[[repl]]$test)
  
  train[, overview := NULL] # model-based boosting cant handle unstructed (i.e. pure text data)
  train <- train[, !grepl("text", colnames(train)), with = FALSE]
  train[, production_countries := NULL] # excluded here since it was excluded when estimated with deepregression()
  train[, vote_count := NULL]
  train[, vote_average := NULL]

  test[, overview := NULL] # model-based boosting cant handle unstructed (i.e. pure text data)
  test <- test[, !grepl("text", colnames(test)), with = FALSE]
  test[, production_countries := NULL] # excluded here since it was excluded when estimated with deepregression()
  test[, vote_count := NULL]
  test[, vote_average := NULL]
  
  ### center variables for appropriate convergence of boosting algo
  # one might think of only de-meaning the numeric and not the dummy coded vars and
  # put intercept = TRUE for the dummy coded bols()
  
  vars <- names(train)[!grepl("revenue", names(train))]
  train[, (vars) := lapply(.SD, scale, center = TRUE, scale = FALSE), .SDcols = vars]
  
  # center test variables around mean of train variables
  means_train <- sapply(train[,..vars], attr, which = "scaled:center")
  for (va_r in names(means_train)) {
    test[[va_r]] <- test[[va_r]] - mean(test[[va_r]]) + means_train[names(means_train) == va_r]
  }
  
  # set up parts of the model with basefun
  y_var <- numeric_var("revenue", support = c(min(c(min(train$revenue),min(train$revenue))), max(c(max(train$revenue),max(train$revenue)))))
  B_y <- Bernstein_basis(y_var, order = 25, ui = "increasing") # 25 was also choosen in movies.R
  
  # start with an unconditional model since thetas are nuissance in stmboost() and betas are nuissance in ctmboost()
  mlt_m <- mlt(ctm(response = B_y, data = train, todistr = "Normal"), data = train)
  
  # intercept = FALSE, since covariates are already centered
  lin_nms <- names(train)[grep("genre", names(train))]
  lin_bl <- paste0("bols(", lin_nms, ", intercept = FALSE)")
  
  p_sp_nms <- names(train)[!grepl("genre|revenue", names(train))]
  p_sp_bl <- paste0("bbs(", p_sp_nms,")")
  
  pred <- paste0(c(p_sp_bl, lin_bl), collapse = " + ")
  mf <- as.formula(paste("revenue ~", pred))
  
  ###### 
  ### STM Boost
  ######
  
  # no offset (such as offset = mean(train_d$revenue)) needed since algo starts with uncond. trafo model
  stm_bm <- stmboost(mlt_m, formula = mf, data = train, method = quote(mboost::gamboost), 
                     control = boost_control(nu = 0.1, mstop = 300, trace = TRUE))
  
  flds <- mboost::cv(weights(mlt_m), B = 10)
  stm_risk <- cvrisk(stm_bm, folds = flds, grid = 1:300)
  # plot(stm_risk)
  stm_m <- stm_bm[mstop(stm_risk)]
  # summary(stm_m)
  
  ## predict density on test set
  
  # lseq <- function(from=1, to=max(d$revenue), length.out=100) {
  #   # logarithmic spaced sequence
  #   # blatantly stolen from library("emdbook"), because need only this
  #   exp(seq(log(from), log(to), length.out = length.out))
  # }
  # 
  # fine_gr <- 100
  # y_gr <- mkgrid(y_var, n = fine_gr)$revenue
  # y_gr <- c(1,y_gr[-1])
  
  y_gr <- sort(test$revenue) # so we can evaluate log-scores exactly
  dens_test <- predict(stm_m, newdata = test, type = "density", q = y_gr)
  
  # Mean predictive (test-set) log-score
  stm_log_sc <- mean(sapply(1:nrow(test), function(idx) log(dens_test[idx,idx])))
  
  ## plot the predictive densities
  # no_densities <- 20
  # col_pal <- rainbow(no_densities)
  # plot(y_gr, dens_test[,1], ylim = c(0, 1e-8), col = col_pal[1], ylab = "density", type = "l", xlab = "revenue")
  # for (idx in 2:no_densities) {
  #   par(new = TRUE)
  #   lines(y_gr, dens_test[,idx], ylim = c(0, 1e-8), col = col_pal[idx], ylab = "", type = "l", xlab = "revenue")
  # }
  
  ###### 
  ### CTM Boost
  ######
  
  # no offset (such as offset = mean(train_d$revenue)) needed since algo starts with uncond. trafo model
  # extension of grid implemented but boosting runs to saturated model (i.e. includes all base learners)
  ctm_bm <- ctmboost(mlt_m, formula = mf, data = train, method = quote(mboost::gamboost), 
                     control = boost_control(nu = 0.1, mstop = 700, trace = TRUE))
  
  flds <- mboost::cv(weights(mlt_m), B = 10)
  ctm_risk <- cvrisk(ctm_bm, folds = flds, grid = 1:700)
  # plot(ctm_risk)
  ctm_m <- ctm_bm[mstop(ctm_risk)]
  # summary(ctm_m)
  
  ## predict density on test set
  # when M = 30 predict() was unable to chol() when coneproj::qprog() was executed
  y_gr <- sort(test$revenue) # so we can evaluate log-scores exactly
  dens_test <- predict(ctm_m, newdata = test, type = "density", q = y_gr)
  
  # Mean predictive (test-set) log-score:
  ctm_log_sc <- mean(sapply(1:nrow(test), function(idx) log(dens_test[idx,idx])))
  
  ## plot the predictive densities
  # no_densities <- 20
  # col_pal <- rainbow(no_densities)
  # plot(y_gr, dens_test[,1], ylim = c(0, 1e-8), col = col_pal[1], ylab = "density", type = "l", xlab = "revenue")
  # for (idx in 2:no_densities) {
  #   par(new = TRUE)
  #   lines(y_gr, dens_test[,idx], ylim = c(0, 1e-8), col = col_pal[idx], ylab = "", type = "l", xlab = "revenue")
  # }
  
  list(ctm_log_sc = ctm_log_sc, ctm_m = ctm_m, stm_log_sc = stm_log_sc, stm_m = stm_m)
  
}, mc.cores = length(data_list))

mean(sapply(res_boost,`[[`,"ctm_log_sc"))
sd(sapply(res_boost,`[[`,"ctm_log_sc"))

mean(sapply(res_boost,`[[`,"stm_log_sc"))
sd(sapply(res_boost,`[[`,"stm_log_sc"))

saveRDS(res_boost, "./application/BoostingRes.RDS")
