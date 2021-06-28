rm(list = ls())
set.seed(1)

library(parallel)
library(mlt)
# library(tensorflow)
# library(tfprobability)
# library(keras)
# try(a <- tfd_normal(0, 1))
B <- 20 # repetitions/cores in parallel

# set wd accordingly

gen_dat <- function() {
  
  # generate data for numerical experiments
  
  n <- 3e3
  p_reg <- 4
  p_sig <- 1
  
  d <- matrix(runif(n * p_reg, -1, 1), ncol = p_reg)
  d <- cbind(d, runif(n, -pi, pi))
  d <- cbind(d, matrix(runif(n * p_sig, 1, 2), ncol = p_sig)) # sigma
  colnames(d) <- paste0("x", 1:(p_reg + p_sig + 1))
  d <- data.frame(d)
  d$samp <- factor("B", levels = c("A", "B"))
  d$samp[sample(1:n, 500)] <- "A" # independent i, take subsample to check small sample properties
  d$x1x6 <- d$x1 * d$x6
  
  # sigma 1,2,3
  sig_1 <- 1
  sig_2 <- 1/d$x6
  #sig_3 <- 1/d$x6^3
  
  # betas
  b1 <- 1
  
  #### g_1 = sinh
  g_1_inv <- function(y) log(y + sqrt(y^2 + 1))
  
  # eta_+: e1 with three diff sigmas
  e1 <- sin(d$x5)
  d$y_g1_e1_s1 <- g_1_inv(e1 + rnorm(n, 0, sig_1))
  d$y_g1_e1_s2 <- g_1_inv(e1 + rnorm(n, 0, sig_2))
  #d$y_g1_e1_s3 <- g_1_inv(e1 + rnorm(n, 0, sig_3))
  
  # eta_+: e2 with three diff sigmas
  e2 <- e1 + b1 * d$x1
  d$y_g1_e2_s1 <- g_1_inv(e2 + rnorm(n, 0, sig_1))
  d$y_g1_e2_s2 <- g_1_inv(e2 + rnorm(n, 0, sig_2))
  #d$y_g1_e2_s3 <- g_1_inv(e2 + rnorm(n, 0, sig_3))
  
  # eta_+: e3 with sigma 2 - estimated by unstructured model specifications only
  e3 <- exp(d$x1 + d$x2 + d$x3 + d$x4) * (1 + sin(d$x5))
  d$y_g1_e3_s2 <- g_1_inv(e3 + rnorm(n, 0, sig_2))
  
  #### g_2 = step function (cf. paper)
  g_2_inv <- function(x) {
    y <- numeric(length(x))
    y[x <= 1] <- exp(x[x <= 1] - 1)
    y[x > 1 & x < 2] <- x[x > 1 & x < 2]
    y[x >= 2] <- log(x[x >= 2] - 1) + 2
    y
  }
  
  # eta_+: e1 with three diff sigmas
  d$y_g2_e1_s1 <- g_2_inv(e1 + rnorm(n, 0, sig_1))
  d$y_g2_e1_s2 <- g_2_inv(e1 + rnorm(n, 0, sig_2))
  #d$y_g2_e1_s3 <- g_2_inv(e1 + rnorm(n, 0, sig_3))
  
  # eta_+: e2 with three diff sigmas
  d$y_g2_e2_s1 <- g_2_inv(e2 + rnorm(n, 0, sig_1))
  d$y_g2_e2_s2 <- g_2_inv(e2 + rnorm(n, 0, sig_2))
  #d$y_g2_e2_s3 <- g_2_inv(e2 + rnorm(n, 0, sig_3))
  
  # eta_+: deep estimation e3 with sigma 2
  d$y_g2_e3_s2 <- g_2_inv(e3 + rnorm(n, 0, sig_2))
  
  d
}

dat <- vector("list", B)
for (b in 1:B) dat[[b]] <- gen_dat()

######
### specifications
######

## h2: shift terms
# bs = "bs" needed for positive entries in model.matrix()

# eta_{+,1}
eta1_sig1 <- as.formula("~ 1 + s(x5, k = 15, bs = 'bs')") # k - 1 columns
eta1_sig2 <- as.formula("~ 1 + s(x5, by = x6, k = 50, bs = 'bs')") # k columns
#eta1_sig3 <- as.formula("~ -1 ")

# eta_{+,2}
eta2_sig1 <- as.formula(" ~ 1 + s(x5, k = 15, bs = 'bs') + x1")
eta2_sig2 <- as.formula("~ 1 + s(x5, by = x6, k = 50, bs = 'bs') + x1x6")
#eta2_sig3 <- as.formula("~ -1 ")

# eta_{+,3} is soley learned by a deep neural net

shift_deep <- function(x) x %>% layer_dense(20, activation = "tanh") %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = 'tanh') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = "tanh") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = "tanh") %>%
  layer_dense(units = 1, activation = "linear")

interact_deep <- function(x) x %>% layer_dense(20, activation = "relu") %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = 'relu') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(20, activation = "relu") %>%
  layer_dense(units = 5, activation = "relu") # relu needed for positive values to retain monotonicity

eta3_sig2_deep <- list("shift_deep" = shift_deep, "interact_deep" = interact_deep)

deep_spec <- list("y_g1_e3_s2", eta3_sig2_deep)

eta3 <- as.formula("~ -1 + shift_deep(x1,x2,x3,x4,x5,x6)") # since we dont know a priori

## h1: interacting term: a(y) rwt sig1/sig2/sig3

# sigma_{+,1}
sig1 <- as.formula('~ 1') # a(y) rwt 1 = a(y)

# sigma_{+,2}
sig2 <- as.formula('~ - 1 + x6')

# deep sigma_{+,2}
sig2_deep <- as.formula("~ -1 + interact_deep(x1,x2,x3,x4,x5,x6)") # since we dont know a priori

# sigma_{+,3}
#sig3 <- as.formula('~ -1 + s(x6, bs = "bs")')

## collect 

specs <- list(
  "g1_e1_s1" = c("y_g1_e1_s1", eta1_sig1, sig1), "g1_e1_s2" = c("y_g1_e1_s2", eta1_sig2, sig2),# "g1_e1_s3" = c("y_g1_e1_s3", eta1_sig3, sig3),
  "g1_e2_s1" = c("y_g1_e2_s1", eta2_sig1, sig1), "g1_e2_s2" = c("y_g1_e2_s2", eta2_sig2, sig2),# "g1_e2_s3" = c("y_g1_e2_s3", eta2_sig3, sig3),
  "g1_e3_s2_deep" = list("y_g1_e3_s2", eta3, sig2_deep, deep_spec),
  "g2_e1_s1" = c("y_g2_e1_s1", eta1_sig1, sig1), "g2_e1_s2" = c("y_g2_e1_s2", eta1_sig2, sig2),# "g2_e1_s3" = c("y_g2_e1_s3", eta1_sig3, sig3),
  "g2_e2_s1" = c("y_g2_e2_s1", eta2_sig1, sig1), "g2_e2_s2" = c("y_g2_e2_s2", eta2_sig2, sig2),# "g2_e2_s3" = c("y_g2_e2_s3", eta2_sig3, sig3),
  "g2_e3_s2_deep" = list("y_g2_e3_s2", eta3, sig2_deep, deep_spec)
)

######
### grid for prediction of interaction term
######

# match with DGP

gr <- 50
x1_var <- numeric_var("x1", support = c(-1, 1))
x2_var <- numeric_var("x2", support = c(-1, 1))
x3_var <- numeric_var("x3", support = c(-1, 1))
x4_var <- numeric_var("x4", support = c(-1, 1))
#x34_var <- numeric_var("x34", support = c(0, 1))
x5_var <- numeric_var("x5", support = c(-pi, pi))
x6_var <- numeric_var("x6", support = c(1, 2)) # sigma
x1x6_var <- numeric_var("x1x6", support = c(-2, 2))
gr_x <- data.frame(mkgrid(x1_var, n = gr), mkgrid(x2_var, n = gr), 
                   mkgrid(x3_var, n = gr), mkgrid(x4_var, n = gr),  
                   mkgrid(x5_var, n = gr), mkgrid(x6_var, n = gr),
                   mkgrid(x1x6_var, n = gr))
#,mkgrid(x34_var, n = gr), mkgrid(x7_var, n = gr))

######
### Run deep estimation
######

run_deep_trafo <- function(d, bsp_M = 15) {
  
  #reticulate::conda_install(envname = "r-reticulate", packages = "gast==0.2.2")
  #tfprobability::install_tfprobability(version=0.8)
  #devtools::load_all("") 
  
  # try(tensorflow:::use_session_with_seed(1)) # works with error
  
  fits <- vector("list") 
  
  for (spec_idx in seq_along(specs)) {
    spec <- specs[[spec_idx]]
    y_var <- spec[[1]]
    
    ifelse(grepl("_g1_",y_var), g <- sinh, g <- function(x) {
      y <- numeric(length(x))
      y[0 < x & x <= 1] <- log(x[0 < x & x <= 1]) + 1
      y[1 < x & x < 2] <- x[1 < x & x < 2]
      y[x >= 2] <- 1 + exp(x[x >= 2] - 2)
      y
    })
    
    cat("Model specification and estimation no.", spec_idx,"\n")
    # only warning: "No deep model specified"
    net_x <- suppressWarnings({deepregression(
      y = d[[y_var]], data = d,
      list_of_formulae = list(spec[[2]], spec[[3]]), # list(h_2_formula, h_1_formula)
      family = "transformation_model",
      list_of_deep_models = if (spec_idx %in% c(5,10)) unlist(spec[[4]][-1], recursive = T) else NULL,
      zero_constraint_for_smooths = TRUE, 
      order_bsp = bsp_M,
      tf_seed = 1,
      addconst_interaction = 0, # adds minimum value (if negative) to the model.matrix of the spline to make all entries positive
      sp_scale = nrow(d),
      df = 8 # penalize spline when too wiggly (uniformly applied to all smooth terms)
    )})
    
    net_x %>% fit(
      epochs = 1e4, validation_split = 0.2, verbose = TRUE, batch_size = nrow(d),
      callbacks = list(
        callback_early_stopping(patience = 200),
        callback_reduce_lr_on_plateau(patience = 100)
      ),
      view_metrics = FALSE
    )
    history <- keras:::to_keras_training_history(net_x$model$history)
    
    # range of y changes with each generated data set, avoids sparse support when adapted in each iteration
    gr_y <- mkgrid(numeric_var("y", support = range(d[[y_var]])), n = nrow(gr_x))
    
    if (!spec_idx %in% c(5,10)) {
      
      # here: no deep model parts are specified (i.e. no unstructured formula)
      
      # evaluate prediction on grid
      # only y and x6 are meaningful here since they appear in the true interaction term: g(y)/sigma(x) (cf. DGP)
      pred_gr <- expand.grid("y" = gr_y$y, "x6" = gr_x$x6)
      nms <- colnames(pred_gr)
      
      # all variables specified in h_1 or h_2 are needed because of prepare_data() 
      # inside predict(). Ought be changed at some point.
      # Using non-meaningful values for x1,x2,x3,x4,x5 since expand.grid() would lead to overlong matrix
      pred_gr <- cbind(pred_gr, matrix(0, nrow = nrow(pred_gr), ncol = 6))
      colnames(pred_gr) <- c(nms, "x1","x2","x3","x4","x5","x1x6")
      
      trf <- net_x %>% predict(pred_gr[, !names(pred_gr) %in% "y"])
      fits[[spec_idx]] <- list("predict" = plot(net_x, plot = FALSE, grid_length = gr, eval_grid = T), # = 50 due to disk capacities on cluster directory
                               "interaction" = cbind(trf(pred_gr$y, type = "interaction")[,c("interaction",all.vars(spec[[3]])), drop = FALSE], y = pred_gr$y),
                               "logLik" = log_score(net_x, summary_fun = sum),
                               "theta" = get_theta(net_x), "shift" = get_shift(net_x),
                               "metrics" = history$metrics,
                               "ident_const" = c(mean(g(d[[y_var]]))) - get_shift(net_x)[1])
    } else {
      
      # only deep model parts are specified (i.e. no structured formula)
      trf <- net_x %>% predict(gr_x)
      fits[[spec_idx]] <- list("h1" = trf(gr_y$y, type = "interaction")[["interaction"]], "h2" = trf(gr_y$y, type = "shift")[["shift"]], 
                               "metrics" = history$metrics, "y" = gr_y$y)
    }
    rm(net_x)
  }
  
  names(fits) <- names(specs)
  
  attr(fits,"seed") <- .Random.seed # may use python RNG stream instead
  attr(fits,"size") <- format(object.size(fits), units = "MB")
  attr(fits,"time") <- Sys.time()
  attr(fits,"session") <- sessionInfo()
  fits
}

cl <- makeCluster(B, outfile = "")
clusterExport(cl, c("specs","gr_x","gr"))
clusterEvalQ(cl,  {library(mlt)
  library(tfprobability)
  library(tensorflow)
  library(keras)})

# full n

res_n_full_M25 <- parLapply(cl, dat, run_deep_trafo, bsp_M = 25)
save(res_n_full_M25, file = "res_n_full_M25.RData")
rm(res_n_full_M25)

res_n_full_M15 <- parLapply(cl, dat, run_deep_trafo, bsp_M = 15)
save(res_n_full_M15, file = "res_n_full_M15.RData")
rm(res_n_full_M15)

# reduced/smaller n

res_n_red_M25 <- parLapply(cl, lapply(dat, function(d) d[d$samp == "A",]), run_deep_trafo, bsp_M = 25)
save(res_n_red_M25, file = "res_n_red_M25.RData")
rm(res_n_red_M25)

res_n_red_M15 <- parLapply(cl, lapply(dat, function(d) d[d$samp == "A",]), run_deep_trafo, bsp_M = 15)
save(res_n_red_M15, file = "res_n_red_M15.RData")
rm(res_n_red_M15)

stopCluster(cl)