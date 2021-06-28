rm(list = ls())
set.seed(1)

library(parallel)
library(tbm)
library(data.table)
B <- 20 # repetitions/no. of cores

# set wd accordingly
# boosting estimates h_2 with a flipped sign


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
  
  data.table(d)
}

dat <- vector("list", B)
for (b in 1:B) dat[[b]] <- gen_dat()
######
### grid for prediction of bbs()
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

y_boost <- c("y_g1_e1_s1", "y_g1_e2_s1", "y_g2_e1_s1", "y_g2_e2_s1")

######
### STMBoost
######

run_stmboost <- function(d, M_BSP = 10){
  
  fits <- vector("list", length(y_boost))
  names(fits) <- y_boost
  
  nms <- names(d)[grepl("^x", names(d))]
  d[, (nms) := lapply(.SD, scale, center = TRUE, scale = FALSE), .SDcols = nms]
  
  # # mean which is taken to de-mean on train, is taken to test since you do not know the test data a priori.
  # # However, we need in-sample predictions rather than out-of-sample predictions here.
  # means_train <- sapply(d[,..nms], attr, which = "scaled:center")
  # for (va_r in names(means_train)) {
  #   gr_x[[va_r]] <- gr_x[[va_r]] - mean(gr_x[[va_r]]) + means_train[names(means_train) == va_r]
  # }
  
  #psp_bls <- paste0("bbs(", nms,")", collapse = " + ")
  #lin_bls <- paste0("bols(", nms,", intercept = FALSE)", collapse = " + ")
  
  B_iter <- 5e2
  
  for (y_nme in y_boost) {
    
    cat("Model estimation:", y_nme,"\n")
    
    yy <- d[[y_nme]]
    #mf <- as.formula(paste(y_nme,"~", paste0(c(psp_bls), collapse = "+")))
    mf <- as.formula(paste(y_nme,"~ bbs(x5)")) # eta_1 and sigma_1
    
    # fair comparison to DCTMs: only specify BL in the way they are generated in 
    # the DGP. We do the same with the s()-terms in deepregression.
    
    bbs_bols <- y_nme %in% c("y_g1_e2_s1","y_g2_e2_s1") # eta_2 and sigma_1
    
    if (bbs_bols) {
      # when same var is bbs() AND bols(), make bbs() center = TRUE, df = 1, to facilitate fair selection compared to bols()
      #lin_bls <- paste0("bols(", nms,", intercept = FALSE)", collapse = " + ")
      #psp_lin <- paste0("bbs(", lin_nms,", center = TRUE, df = 1)", collapse = " + ")
      #psp_bls <- paste0("bbs(", nms[!nms %in% lin_nms],")", collapse = " + ")
      #mf <- as.formula(paste(y_nme,"~", paste0(c(lin_bls), collapse = "+")))
      mf <- as.formula(paste(y_nme,"~  bols(x1, intercept = FALSE) + bbs(x5)"))
    }
    
    # y support changes in each draw
    y_var <- numeric_var(y_nme, support = range(yy))
    B_y <- Bernstein_basis(y_var, order = M_BSP, ui = "increasing")
    
    # start with an unconditional model
    mlt_m <- mlt(ctm(response = B_y, data = d, todistr = "Normal"), data = d)
    
    # no offset (e.g. offset = mean(d$y)) needed since algo starts with uncond. trafo model
    stm_bm <- stmboost(mlt_m, formula = mf, data = d, method = quote(mboost::gamboost),
                       control = boost_control(nu = 0.1, mstop = B_iter, trace = TRUE))
    
    flds <- mboost::cv(weights(mlt_m), B = 5)
    stm_risk <- cvrisk(stm_bm, folds = flds, grid = 1:B_iter)
    # plot(ctm_risk)
    stm_bm <- stm_bm[mstop(stm_risk)]
    
    # h_2: coefs for bols() and predictions for bbs()
    #bbs_pred <- bols_coef <- vector("list")
    #for (nm in nms) bols_coef[[nm]] <- mboost:::coef.mboost(stm_bm, which = nm)
    #for (nm in nms) bbs_pred[[nm]] <- list("x" = gr_x[[nm]] ,"prd" = predict.mboost(stm_bm, newdata = gr_x, which = nm))
    bols_coef <- ifelse(bbs_bols, mboost:::coef.mboost(stm_bm, which = "x1"), NA)
    bbs_pred <- list("x" = gr_x[["x5"]] ,"prd" = predict.mboost(stm_bm, newdata = gr_x, which = "x5"))
    
    # h_1: thetas are nuissance, however, updated in each boosting iteration
    h1_hat <- model.matrix(B_y, data = data.frame(y_nme = mkgrid(y_var, n = 50))) %*% coef(stm_bm)[1,1:(M_BSP + 1)]
    fits[[y_nme]] <- list("bbs_pred" = bbs_pred, "bols_coef" = bols_coef, "h1_hat" = h1_hat, "y_gr" = mkgrid(y_var, n = 50))
  }
  
  fits
}

cl <- makeCluster(20, outfile = "")
clusterEvalQ(cl, {library(tbm)
  library(data.table)})
clusterExport(cl, c("y_boost","gr_x"))

# full n

res_n_full_M25 <- parLapply(cl, dat, run_stmboost, M_BSP = 25)
save(res_n_full_M25, file = "boost_res_n_full_M25.RData")
rm(res_n_full_M25)

res_n_full_M15 <- parLapply(cl, dat, run_stmboost, M_BSP = 15)
save(res_n_full_M15, file = "boost_res_n_full_M15.RData")
rm(res_n_full_M15)

# reduced/smaller n

res_n_red_M25 <- parLapply(cl, lapply(dat, function(d) d[d$samp == "A",]), run_stmboost, M_BSP = 25)
save(res_n_red_M25, file = "boost_res_n_red_M25.RData")
rm(res_n_red_M25)

res_n_red_M15 <- parLapply(cl, lapply(dat, function(d) d[d$samp == "A",]), run_stmboost, M_BSP = 15)
save(res_n_red_M15, file = "boost_res_n_red_M15.RData")
rm(res_n_red_M15)

stopCluster(cl)