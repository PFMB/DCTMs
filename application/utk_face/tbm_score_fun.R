stm_ctm_boost_eval <- function(train_x, train_y, test_x, test_y, 
                               family = "Logistic", Bsp_order = 25,
                               lin_names = NULL,
                               bbs_names = NULL, outcome_name = "y",
                               scale_feature_names = colnames(train_x),
                               control = boost_control(nu = 0.3, mstop = 5e2, trace = TRUE),
                               B = 3)
{
  
  vars <- scale_feature_names
  train_x <- as.data.table(train_x)
  train_x[, (vars) := lapply(.SD, scale, center = TRUE, scale = FALSE), 
          .SDcols = vars]
  
  # center test variables around mean of train variables
  means_train <- sapply(train_x[,..vars], attr, which = "scaled:center")
  for (va_r in names(means_train)) {
    test_x[[va_r]] <- test_x[[va_r]] - mean(test_x[[va_r]]) + 
      means_train[names(means_train) == va_r]
  }
  
  # set up parts of the model with basefun
  y_var <- numeric_var(outcome_name, 
                       support = c(min(c(train_y,test_y)), 
                                   max(c(train_y,test_y))))
  B_y <- Bernstein_basis(y_var, order = Bsp_order, ui = "increasing") 
  # 25 was also choosen in movies.R
  
  train_x <- cbind(train_y, train_x)
  colnames(train_x)[1] <- outcome_name
  
  # start with an unconditional model since thetas are nuissance in stmboost() 
  # and betas are nuissance in ctmboost()
  mlt_m <- mlt(ctm(response = B_y, 
                   data = train_x, 
                   todistr = family), 
               data = train_x)
  
  # intercept = FALSE, since covariates are already centered
  lin_bl <- p_sp_bl <- NULL
  
  if(!is.null(lin_names))
    lin_bl <- paste0("bols(", lin_names, ", intercept = FALSE)")
  
  if(!is.null(bbs_names))
    p_sp_bl <- paste0("bbs(", bbs_names,")")
  
  pred <- paste0(c(p_sp_bl, lin_bl), collapse = " + ")
  mf <- as.formula(paste(outcome_name," ~", pred))
  
  ###### 
  ### STM Boost
  ######

  # # no offset (such as offset = mean(train_d$revenue)) needed since algo starts with uncond. trafo model
  # stm_bm <- tryCatch({stm_bm <- stmboost(mlt_m,
  #                    data = train_x,
  #                    formula = mf,
  #                    method = quote(mboost::gamboost),
  #                    control = control)
  # 
  #   stm_risk <- cvrisk(stm_bm, folds = cv(model.weights(stm_bm), type = "kfold", B = B), papply = lapply())
  #   # plot(stm_risk)
  #   stm_bm[mstop(stm_risk)]
  # }, error = function(x) return(NA))

  y_gr <- sort(test_y) # so we can evaluate log-scores exactly

  # res <- list("stm_bm" = stm_bm, "y_gr" = y_gr, "test_x" = test_x)
  # options(digits.secs = 6) # due to parallel saving
  # save(res, file = paste0("res_fit_stm_",Sys.time()))
  # 
  # stm_log_sc <- tryCatch({dens_test <- predict(stm_bm, newdata = test_x, type = "density", q = y_gr)
  # 
  #   # Mean predictive (test-set) log-score
  #   mean(sapply(1:nrow(test_x), function(idx) log(dens_test[idx,idx])))
  # }, error = function(x) return(NA))
  
  ###### 
  ### CTM Boost
  ######
  
  # no offset (such as offset = mean(train_d$revenue)) needed since algo starts with uncond. trafo model
  # extension of grid implemented but boosting runs to saturated model (i.e. includes all base learners)
  ctm_bm <- ctmboost(mlt_m, 
                     data = train_x,
                     formula = mf, 
                     method = quote(mboost::gamboost), 
                     control = control)
  
  cat("Checkpoint 1 \n")
  
  ctm_risk <- cvrisk(ctm_bm, folds = cv(model.weights(ctm_bm), type = "kfold", B = B), papply = lapply)
  # plot(ctm_risk)
    
  cat("Checkpoint 2 \n")
  ctm_bm <- ctm_bm[mstop(ctm_risk)]
  
  cat("Checkpoint 3 \n")
  
  res <- list("ctm_bm" = ctm_bm, "y_gr" = y_gr, "test_x" = test_x)
  options(digits.secs = 6) # due to parallel saving
  save(res, file = paste0("res_fit_ctm_",Sys.time()))
  # dens_test <- predict(stm_bm, newdata = test_x, 
  #                      type = "density", q = y_gr)
    
  ctm_log_sc <- tryCatch({dens_test <- predict(ctm_bm, newdata = test_x, type = "density", q = y_gr)
  
    # Mean predictive (test-set) log-score:
    mean(sapply(1:nrow(test_x), function(idx) log(dens_test[idx,idx])))
  }, error = function(x) return(NA))
  
  return(list(stm_score = NA,
              ctm_score = ctm_log_sc))
  
}

