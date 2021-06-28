rm(list = ls())
library(ggplot2)
library(ggsci) # journal style plot colors
library(data.table)
library(tidyverse)
library(basefun)
library(metR)
library(cowplot)

# set wd accordingly

# measure for evaluation
relimse <- function(prd, trth) sum(c(((prd - trth)^2)), na.rm = T) / sum(c(trth^2), na.rm = T)
mse <- function(prd, trth) sum((prd - trth)^2)/length(prd)

# load results of numerical experiments
net_res <- list.files(pattern = "^res_n_")
net_res <- sapply(net_res, load, envir = .GlobalEnv)

reps <- 20 # set number of repetitions

#######
### True functional form, values
######

true_coef <- c("x1" = 1)

g1 <- function(x) sinh(x)
g2 <- function(x) {
  y <- numeric(length(x))
  y[0 < x & x <= 1] <- log(x[0 < x & x <= 1]) + 1
  y[1 < x & x < 2] <- x[1 < x & x < 2]
  y[x >= 2] <- 1 + exp(x[x >= 2] - 2)
  y
}

# match with DGP
# for h_2 we use the in-sample grid 
# for h_1 we use a ordered grid

gr <- 50
x1_var <- numeric_var("x1", support = c(-1, 1))
x2_var <- numeric_var("x2", support = c(-1, 1))
x3_var <- numeric_var("x3", support = c(-1, 1))
x4_var <- numeric_var("x4", support = c(-1, 1))
x5_var <- numeric_var("x5", support = c(-pi, pi))
x6_var <- numeric_var("x6", support = c(1, 2))
x1x6_var <- numeric_var("x1x6", support = c(-2, 2))
gr_x <- data.frame(mkgrid(x1_var, n = gr), mkgrid(x2_var, n = gr), 
                   mkgrid(x3_var, n = gr), mkgrid(x4_var, n = gr),  
                   mkgrid(x5_var, n = gr), mkgrid(x6_var, n = gr),
                   mkgrid(x1x6_var, n = gr))

#######
### Calculate MSE/RIMSE
######

#rm(list = setdiff(ls(), c("g","func_nm","d_set","res_n_full_M25","gr_x","relimse","mse","true_coef","reps")))

get_res <- function(d_set, g = function(x){x}, func_nm = "g1") {
  
  #### g_1, eta_1, sigma_1
  g_e1_s1 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e1_s1"))
  
  # h1
  h1_g_e1_s1 <- lapply(g_e1_s1, `[[`, "interaction")
  ident_const <- lapply(g_e1_s1,`[[`, "ident_const")
  h1_pr <- lapply(h1_g_e1_s1, `[[`, "interaction")
  h1_pr <- lapply(h1_pr, function(x) x[1:50]) # 2500 times the same vector
  h1_pr <- lapply(1:reps, function(idx) h1_pr[[idx]] + ident_const[[idx]])
  h1_gr <- lapply(h1_g_e1_s1, `[[`, "y")
  h1_tr <- lapply(h1_gr, function(x) g(x[1:50]))
  h1_g_e1_s1 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # plot(h1_tr[[1]], ylim = c(-5,5))
  # par(new = TRUE)
  # plot(h1_pr[[1]], ylim = c(-5,5), col = "red")
  
  # h2
  h2_g_e1_s1 <- lapply(g_e1_s1, `[[`, "predict")
  h2_g_e1_s1 <- lapply(h2_g_e1_s1, `[[`, c(1))
  x5_grid <- lapply(h2_g_e1_s1, `[[`, "value")
  pr_sin_x5 <- lapply(h2_g_e1_s1, `[[`, "partial_effects")
  tr_sin_x5 <- lapply(x5_grid, function(gr) sin(gr))
  h2_g_e1_s1 <- sapply(1:reps, function(idx) relimse(pr_sin_x5[[idx]], tr_sin_x5[[idx]]))
  # plot(x5_grid[[1]],pr_sin_x5[[1]], ylim = c(-1,1))
  # par(new = TRUE)
  # plot(x5_grid[[1]],tr_sin_x5[[1]], ylim = c(-1,1), col = "red")
  
  #### g_1, eta_1, sigma_2
  g_e1_s2 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e1_s2"))
  
  # h1
  h1_g_e1_s2 <- lapply(g_e1_s2, `[[`, "interaction")
  ident_const <- lapply(g_e1_s2,`[[`, "ident_const")
  h1_pr <- lapply(h1_g_e1_s2, `[[`, "interaction")
  h1_pr <- lapply(1:reps, function(idx) h1_pr[[idx]] + ident_const[[idx]])
  h1_tr <- lapply(h1_g_e1_s2, function(gr) g(gr$y) * gr$x6)
  # gr <- h1_g_e1_s2[[1]]
  h1_g_e1_s2 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # par(mfrow = c(1,2))
  # image(unique(gr$x6),unique(gr$y), matrix(h1_pr[[1]], ncol = 50))
  # contour(unique(gr$x6),unique(gr$y), matrix(h1_pr[[1]], ncol = 50), add = T)
  # image(unique(gr$x6),unique(gr$y), matrix(h1_tr[[1]], ncol = 50))
  # contour(unique(gr$x6),unique(gr$y), matrix(h1_tr[[1]], ncol = 50), add = T)
  
  # h2
  h2_g_e1_s2 <- lapply(g_e1_s2, `[[`, "predict")
  h2_g_e1_s2 <- lapply(h2_g_e1_s2, `[[`, c(1))
  # gr <- lapply(h2_g_e1_s2, `[[`, "df")[[1]]
  h2_tr <- lapply(h2_g_e1_s2, function(x) matrix(sin(x$df$x5) * x$df$x6, ncol = 50))
  h2_pr <- lapply(h2_g_e1_s2, function(x) matrix(x$pred, ncol = 50))
  h2_g_e1_s2 <- sapply(1:reps, function(idx) relimse(h2_pr[[idx]], h2_tr[[idx]]))
  # par(mfrow = c(1,2))
  # image(unique(gr$x5),unique(gr$x6), matrix(h2_pr[[1]], ncol = 50))
  # contour(unique(gr$x5),unique(gr$x6), matrix(h2_pr[[1]], ncol = 50), add = T)
  # image(unique(gr$x5),unique(gr$x6), matrix(h2_tr[[1]], ncol = 50))
  # contour(unique(gr$x5),unique(gr$x6), matrix(h2_tr[[1]], ncol = 50), add = T)
  
  # #### g_1, eta_1, sigma_3 
  # g_e1_s3 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e1_s3"))
  # 
  # # h1
  # h1_g_e1_s3 <- lapply(g_e1_s3, `[[`, "interaction")
  # h1_pr <- lapply(h1_g_e1_s3, `[[`, "interaction")
  # h1_pr <- lapply(h1_pr, function(x) x[1:2500]) # 50 times the same vector
  # h1_tr <- lapply(h1_g_e1_s3, function(gr) {
  #   gr <- gr[1:2500,] # 50 times the same vector
  #   g(gr$y) * gr$x6^3
  # })
  # h1_g_e1_s3 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # 
  # # h2
  # h2_g_e1_s3 <- lapply(g_e1_s3, `[[`, "predict")
  # h2_g_e1_s3  <- lapply(1:length(true_coef), function(regr) {
  #   reg_x6 <- lapply(h2_g_e1_s3, `[[`, c(regr))
  #   gr_reg_x6 <- lapply(reg_x6, `[[`, "df")
  #   pr_reg_x6 <- lapply(reg_x6, `[[`, "pred")
  #   regr_nm <- names(true_coef[regr])
  #   tr_reg_x6 <- lapply(gr_reg_x6, function(gr) true_coef[regr_nm]*gr[[regr_nm]] * gr[["x6"]]^3)
  #   sapply(1:reps, function(idx) relimse(pr_reg_x6[[idx]], tr_reg_x6[[idx]]))
  # })
  # h2_g_e1_s3 <- Reduce("+",h2_g_e1_s3)/3 # Average taken for 3 model terms in e1/s2
  
  #### g_1, eta_2, sigma_1
  g_e2_s1 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e2_s1"))
  
  # h1
  ident_const <- lapply(g_e2_s1,`[[`, "ident_const")
  h1_g_e2_s1 <- lapply(g_e2_s1, `[[`, "interaction")
  h1_pr <- lapply(h1_g_e2_s1, `[[`, "interaction")
  h1_pr <- lapply(h1_pr, function(x) x[1:50]) # 50 different realizations
  h1_pr <- lapply(1:reps, function(idx) h1_pr[[idx]] + ident_const[[idx]])
  h1_gr <- lapply(h1_g_e2_s1, `[[`, "y")
  h1_tr <- lapply(h1_gr, function(x) g(x[1:50]))
  h1_g_e2_s1 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # plot(h1_tr[[1]], ylim = c(-5,5))
  # par(new = TRUE)
  # plot(h1_pr[[1]], ylim = c(-5,5), col = "red")
  
  # h2 sin(x) + \beta x
  h2_g_e2_s1 <- lapply(g_e2_s1, `[[`, "predict")
  h2_g_e2_s1 <- do.call("c",h2_g_e2_s1)
  gr_g_e2_s1 <- lapply(h2_g_e2_s1, `[[`, "value")
  tr_g_e2_s1 <- lapply(gr_g_e2_s1, function(x) sin(x))
  pr_g_e2_s1 <- lapply(h2_g_e2_s1, `[[`, "partial_effects")
  h2_g_e2_s1 <- sapply(1:reps, function(idx) relimse(pr_g_e2_s1[[idx]], tr_g_e2_s1[[idx]]))
  # plot(gr_g_e2_s1[[1]],tr_g_e2_s1[[1]], ylim = c(-1,1))
  # par(new = TRUE)
  # plot(gr_g_e2_s1[[1]],pr_g_e2_s1[[1]], ylim = c(-1,1), col = "red")
  
  # \beta x
  h2_beta <- lapply(g_e2_s1, `[[`, "shift")
  beta <- sapply(h2_beta, function(r) r[2])
  mse_beta <- mean((beta - 1)^2)
  h2_g_e2_s1 <- h2_g_e2_s1 + mse_beta
  
  #### g_1, eta_2, sigma_2
  g_e2_s2 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e2_s2"))
  
  # h1
  ident_const <- lapply(g_e2_s2,`[[`, "ident_const")
  h1_g_e2_s2 <- lapply(g_e2_s2, `[[`, "interaction")
  # gr <- h1_g_e1_s2[[1]]
  h1_pr <- lapply(h1_g_e2_s2, `[[`, "interaction")
  h1_pr <- lapply(1:reps, function(idx) h1_pr[[idx]] + ident_const[[idx]])
  h1_tr <- lapply(h1_g_e2_s2, function(gr) g(gr$y) * gr$x6)
  h1_g_e2_s2 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # par(mfrow = c(1,2))
  # image(unique(gr$x6),unique(gr$y), matrix(h1_pr[[1]], ncol = 50))
  # contour(unique(gr$x6),unique(gr$y), matrix(h1_pr[[1]], ncol = 50), add = T)
  # image(unique(gr$x6),unique(gr$y), matrix(h1_tr[[1]], ncol = 50))
  # contour(unique(gr$x6),unique(gr$y), matrix(h1_tr[[1]], ncol = 50), add = T)
  
  # h2
  h2_g_e2_s2 <- lapply(g_e2_s2, `[[`, "predict")
  h2_g_e2_s2 <- do.call("c",h2_g_e2_s2)
  gr <- h2_g_e2_s2[[1]]$df
  pr_x5_x6 <- lapply(h2_g_e2_s2, `[[`, "pred")
  pr_x5_x6 <- lapply(pr_x5_x6, function(x) matrix(x, ncol = 50))
  tr_x5_x6 <- lapply(h2_g_e2_s2, function(gr) matrix(sin(gr$df$x5) * gr$df$x6, ncol = 50))
  h2_g_e2_s2 <- sapply(1:reps, function(idx) relimse(pr_x5_x6[[idx]], tr_x5_x6[[idx]]))
  # par(mfrow = c(1,2))
  # image(unique(gr$x5),unique(gr$x6), matrix(pr_x5_x6[[1]], ncol = 50))
  # contour(unique(gr$x5),unique(gr$x6), matrix(pr_x5_x6[[1]], ncol = 50), add = T)
  # image(unique(gr$x5),unique(gr$x6), matrix(tr_x5_x6[[1]], ncol = 50))
  # contour(unique(gr$x5),unique(gr$x6), matrix(tr_x5_x6[[1]], ncol = 50), add = T)
  
  # \beta x1 * x6
  h2_beta <- lapply(g_e2_s2, `[[`, "shift")
  beta <- sapply(h2_beta, function(r) r[2])
  mse_beta <- mean((beta - 1)^2)
  h2_g_e2_s2 <- h2_g_e2_s2 + mse_beta
  
  # #### g_1, eta_2, sigma_3
  # g_e2_s3 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e2_s3"))
  # 
  # # h1
  # h1_g_e2_s3 <- lapply(g_e2_s3, `[[`, "interaction")
  # h1_pr <- lapply(h1_g_e2_s3, `[[`, "interaction")
  # h1_pr <- lapply(h1_pr, function(x) x[1:2500]) # 50 times the same vector
  # h1_tr <- lapply(h1_g_e2_s3, function(gr) {
  #   gr <- gr[1:2500,]
  #   g(gr$y) * gr$x6^3
  # })
  # h1_g_e2_s3 <- sapply(1:reps, function(idx) relimse(h1_pr[[idx]], h1_tr[[idx]]))
  # 
  # # h2
  # h2_g_e2_s3 <- lapply(g_e2_s3, `[[`, "predict")
  # h2_g_e2_s3 <- do.call("c",h2_g_e2_s3)
  # gr_g_x5_x6 <- lapply(h2_g_e2_s3, `[[`, "df")
  # pr_g_x5_x6 <- lapply(h2_g_e2_s3, `[[`, "pred")
  # tr_g_x5_x6 <- lapply(gr_g_x5_x6, function(gr) sin(gr$x5) * gr$x6^3)
  # h2_g_e2_s3 <- sapply(1:reps, function(idx) relimse(pr_g_x5_x6[[idx]], tr_g_x5_x6[[idx]]))
  
  #### g_1, eta_3, sigma_2 only deep estimation
  g_e3_s2 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e3_s2_deep"))
  
  # h1
  pr_g_e3_s2 <- lapply(g_e3_s2, `[[`, "h1")
  y_gr_e3_s2 <- lapply(g_e3_s2, `[[`, "y")
  tr_g_e3_s2 <- lapply(y_gr_e3_s2, function(y_gr) g(y_gr) * gr_x$x6)
  h1_g_e3_s2 <- sapply(1:reps, function(idx) relimse(pr_g_e3_s2[[idx]], tr_g_e3_s2[[idx]]))
  
  # h2
  pr_g_e3_s2 <- lapply(g_e3_s2, `[[`, "h2")
  tr_g_e3_s2 <- exp(gr_x$x1 + gr_x$x2 + gr_x$x3 + gr_x$x4) * (1 + sin(gr_x$x5)) * gr_x$x6
  tr_g_e3_s2 <- list(tr_g_e3_s2)[rep(1,reps)] # same grid as for the prediction of h2, see RunSim.R
  h2_g_e3_s2 <- sapply(1:reps, function(idx) relimse(pr_g_e3_s2[[idx]], tr_g_e3_s2[[idx]]))
  
  res <- list(h1_g_e1_s1, h2_g_e1_s1, h1_g_e1_s2, h2_g_e1_s2, 
              h1_g_e2_s1, h2_g_e2_s1, h1_g_e2_s2, h2_g_e2_s2,
              h1_g_e3_s2, h2_g_e3_s2)
  
  setNames(res, c(paste0("h1_",func_nm,"_e1_s1"), paste0("h2_",func_nm,"_e1_s1"), paste0("h1_",func_nm,"_e1_s2"), paste0("h2_",func_nm,"_e1_s2"),
                  paste0("h1_",func_nm,"_e2_s1"), paste0("h2_",func_nm,"_e2_s1"), paste0("h1_",func_nm,"_e2_s2"), paste0("h2_",func_nm,"_e2_s2"),
                  paste0("h1_",func_nm,"_e3_s2"), paste0("h2_",func_nm,"_e3_s2")))
}

res_g1_full_M15 <- get_res("res_n_full_M15", g = g1, func_nm = "g1")
res_g2_full_M15 <- get_res("res_n_full_M15", g = g2, func_nm = "g2")
res_g1_full_M25 <- get_res("res_n_full_M25", g = g1, func_nm = "g1")
res_g2_full_M25 <- get_res("res_n_full_M25", g = g2, func_nm = "g2")

res_g1_red_M15 <- get_res("res_n_red_M15", g = g1, func_nm = "g1")
res_g2_red_M15 <- get_res("res_n_red_M15", g = g2, func_nm = "g2")
res_g1_red_M25 <- get_res("res_n_red_M25", g = g1, func_nm = "g1")
res_g2_red_M25 <- get_res("res_n_red_M25", g = g2, func_nm = "g2")

##### collect

full_M15 <- data.table(cbind(do.call("cbind",res_g1_full_M15), do.call("cbind",res_g2_full_M15)))
h1_full_M15 <- full_M15[,grep("^h1",colnames(full_M15)), with = FALSE]
h2_full_M15 <- full_M15[,grep("^h2",colnames(full_M15)), with = FALSE]

full_M25 <- data.table(cbind(do.call("cbind",res_g1_full_M25), do.call("cbind",res_g2_full_M25)))
h1_full_M25 <- full_M25[,grep("^h1",colnames(full_M25)), with = FALSE]
h2_full_M25 <- full_M25[,grep("^h2",colnames(full_M25)), with = FALSE]

red_M15 <- data.table(cbind(do.call("cbind",res_g1_red_M15), do.call("cbind",res_g2_red_M15)))
h1_red_M15 <- red_M15[,grep("^h1",colnames(red_M15)), with = FALSE]
h2_red_M15 <- red_M15[,grep("^h2",colnames(red_M15)), with = FALSE]

red_M25 <- data.table(cbind(do.call("cbind",res_g1_red_M25), do.call("cbind",res_g2_red_M25)))
h1_red_M25 <- red_M25[,grep("^h1",colnames(red_M25)), with = FALSE]
h2_red_M25 <- red_M25[,grep("^h2",colnames(red_M25)), with = FALSE]

#### h1

h1 <- list(red_M15 = h1_red_M15, red_M25 = h1_red_M25, full_M15 = h1_full_M15, full_M25 = h1_full_M25)

sett <- cbind(rep(c("M = 15","M = 25"),2),
              c("n = 500","n = 500", "n = 3000","n = 3000"))
for (idx in 1:length(h1)) {
  d <- h1[[idx]]
  d1 <- data.table(matrix(rep(sett[idx,], reps), ncol = 2, byrow = TRUE))
  setnames(d1, c("V1","V2"), c("BSP Order", "Sample size"))
  h1[[idx]] <- cbind(d, d1)
}

h1 <- rbindlist(h1)
h1 <- melt(h1, id.vars = c("BSP Order","Sample size"), variable.name = "DGP", value.name = "(RI)MSE")
h1[, `BSP Order` := factor(`BSP Order`)]
h1[, `Sample size` := factor(`Sample size`, levels = c("n = 500", "n = 3000"))]

# labeller = label_parsed changes column as well as row labels, even if we only want to change the rows
h1[, `Sample size` := factor(`Sample size`, labels = c(expression(n == 500), expression(n == 3000)))]

lbl <- c(expression(g[1] * eta[1] * sigma[1]),
         expression(g[1] * eta[1] * sigma[2]),
         #expression(g[1] * eta[1] * sigma[3]),
         
         expression(g[1] * eta[2] * sigma[1]),
         expression(g[1] * eta[2] * sigma[2]),
         #expression(g[1] * eta[2] * sigma[3]),
         
         expression(g[1] * eta[3] * sigma[2]),
         
         expression(g[2] * eta[1] * sigma[1]),
         expression(g[2] * eta[1] * sigma[2]),
         #expression(g[2] * eta[1] * sigma[3]),
         
         expression(g[2] * eta[2] * sigma[1]),
         expression(g[2] * eta[2] * sigma[2]),
         #expression(g[2] * eta[2] * sigma[3]),
         
         expression(g[2] * eta[3] * sigma[2]))

h1[, DGP := factor(DGP, labels = lbl)]
save(h1, file = "h1_res_dctm_ECML.RData")

(gg <- h1 %>% ggplot(aes(x = `BSP Order`, y = `(RI)MSE`, fill = DGP)) +
    geom_boxplot() + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 16), legend.position = "none"))

#### h2

h2 <- list(red_M15 = h2_red_M15, red_M25 = h2_red_M25, full_M15 = h2_full_M15, full_M25 = h2_full_M25)

for (idx in 1:length(h2)) {
  d <- h2[[idx]]
  d1 <- data.table(matrix(rep(sett[idx,], reps), ncol = 2, byrow = TRUE))
  setnames(d1, c("V1","V2"), c("BSP Order", "Sample size"))
  h2[[idx]] <- cbind(d, d1)
}

h2 <- rbindlist(h2)
h2 <- melt(h2, id.vars = c("BSP Order","Sample size"), variable.name = "DGP", value.name = "(RI)MSE")
h2[, `BSP Order` := factor(`BSP Order`)]
h2[, `Sample size` := factor(`Sample size`, levels = c("n = 500", "n = 3000"))]

# labeller = label_parsed changes column as well as row labels, even if we only want to change the rows
h2[, `Sample size` := factor(`Sample size`, labels = c(expression(n == 500), expression(n == 3000)))]

h2[, DGP := factor(DGP, labels = lbl)]
save(h2, file = "h2_res_dctm_ECML.RData")

(gg <- h2 %>% ggplot(aes(x = `BSP Order`, y = `(RI)MSE`, fill = DGP)) +
    geom_boxplot() + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 16), legend.position = "top"))

#### Figure 3: Exemplary visualization of the learned feature-driven interaction term h_1

h1_figure <- h1[DGP == "g[1] * eta[1] * sigma[2]" & `BSP Order` == "M = 15" & `Sample size` == "n == 3000" ,]
#h1_figure[order(`(RI)MSE`),]

graphics.off()
dev.off()
par(mfrow = c(2,2))
par(mar = c(4, 4.5, 2.5, 1))
par(cex.lab = 1.8, cex.axis = 1.8, cex.main = 1.8, cex.sub = 1.8)

# setting: g_1, eta_1, sigma_2 

d_set <- "res_n_full_M15"
func_nm <- "g1" # sinh
g <- g1
g_e1_s2 <- lapply(eval(parse(text = d_set)), `[[`, paste0(func_nm, "_e1_s2"))

# h1 = sinh(y) * x_6
h1_g_e1_s2 <- lapply(g_e1_s2, `[[`, "interaction")
h1_c1 <- lapply(g_e1_s2, `[[`, "ident_const")
h1_pr <- lapply(h1_g_e1_s2, `[[`, "interaction")
h1_pr <- lapply(1:reps, function(idx) h1_pr[[idx]] + h1_c1[[idx]])
h1_tr <- lapply(h1_g_e1_s2, function(gr) { # same across all reps
  g(gr$y) * gr$x6
})

gr_h1 <- h1_g_e1_s2[[1]] # plotting grid
tr <- matrix(h1_tr[[1]], ncol = 50)
image(unique(gr_h1$x6), unique(gr_h1$y), tr, ylab = "y", xlab = expression(x[6]), main = expression(h[1]))
contour(unique(gr_h1$x6), unique(gr_h1$y), tr, add = T)
image(unique(gr_h1$x6), unique(gr_h1$y), matrix(h1_pr[[2]],ncol = 50), ylab = "y", xlab = expression(x[6]), main = expression(hat(h[1])))
contour(unique(gr_h1$x6), unique(gr_h1$y), matrix(h1_pr[[2]],ncol = 50), add = T)

# h2 = sin(x_5) * x_6
h2_g_e1_s2 <- lapply(g_e1_s2, `[[`, "predict")
reg_x6 <- lapply(h2_g_e1_s2, `[[`, c(1))
gr_reg_x6 <- lapply(reg_x6, `[[`, "df")
pr_reg_x6 <- lapply(reg_x6, `[[`, "pred")
tr_reg_x6 <- lapply(gr_reg_x6, function(gr) sin(gr[["x5"]]) * gr$x6)

pr <- matrix(pr_reg_x6[[1]], ncol = 50)
tr <- matrix(tr_reg_x6[[1]], ncol = 50)

gr_h2 <- gr_reg_x6[[1]]
image(unique(gr_h2$x5), unique(gr_h2$x6), tr, ylab = expression(x[6]), xlab = expression(x[5]), main = expression(h[2]))
contour(unique(gr_h2$x5), unique(gr_h2$x6), tr, add = T)
image(unique(gr_h2$x5), unique(gr_h2$x6), pr, ylab = expression(x[6]), xlab = expression(x[5]), main = expression(hat(h[2])))
contour(unique(gr_h2$x5), unique(gr_h2$x6),pr, add = T)

## gg plot version
detach("package:basefun",unload=TRUE) # conflict: plot.margin "unit" also in "variables"
detach("package:variables",unload=TRUE)
idx <- 1
graphics.off()
dev.off()
par(mfrow = c(2,2))
h1_true <- data.frame("h1" = h1_tr[[idx]], "y" = gr_h1$y, "x_6" = gr_h1$x6)
h1_pred <- data.frame("h1" = h1_pr[[idx]], "y" = gr_h1$y, "x_6" = gr_h1$x6)

br <- range(c(h1_true,h1_pred))
br <- seq(br[1], br[2], length.out = 30)
(p1 <- ggplot(h1_true, aes(y, x_6, z = h1)) + geom_contour_fill(aes(z = h1), breaks = br) + 
    theme_light() + scale_x_continuous(expand = c(0,0)) + scale_y_continuous(expand = c(0,0)) +
    geom_contour(color = "black") + theme(text = element_text(size = 12), legend.position = "none") + 
    geom_text_contour(label.placement = label_placement_n(1), 
                      rotate = FALSE, stroke = 0.1, skip = 0) + xlab(expression(y)) + ylab(expression(x[6])) + 
    scale_fill_divergent(low ="#67a9cf", mid = "#f7f7f7", high = "#ef8a62")  + labs(title = expression(h[1])) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(plot.margin = unit(c(0.01,0.1,0.1,0.1), "cm"))) #t,r,b,l

(p2 <- ggplot(h1_pred, aes(y, x_6, z = h1)) + geom_contour_fill(aes(z = h1), breaks = br) + 
    theme_light() + scale_x_continuous(expand = c(0,0)) + scale_y_continuous(expand = c(0,0)) +
    geom_contour(color = "black") + theme(text = element_text(size = 12), legend.position = "none") + 
    geom_text_contour(label.placement = label_placement_n(1), 
                      rotate = FALSE, stroke = 0.1, skip = 0) + xlab(expression(y)) + ylab(expression(x[6])) + 
    scale_fill_divergent(low ="#67a9cf", mid = "#f7f7f7", high = "#ef8a62") + labs(title = expression(hat(h[1]))) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(plot.margin = unit(c(0.01,0.1,0.1,0.1), "cm"))) #t,r,b,l

h2_true <- data.frame("h2" = tr_reg_x6[[idx]], "x_5" = gr_h2$x5, "x_6" = gr_h2$x6)
h2_pred <- data.frame("h2" = pr_reg_x6[[idx]], "x_5" = gr_h2$x5, "x_6" = gr_h2$x6)

br <- range(c(h2_true,h2_pred))
br <- seq(br[1], br[2], length.out = 30)
(p3 <- ggplot(h2_true, aes(x_5, x_6, z = h2)) + geom_contour_fill(aes(z = h2), breaks = br) + 
    theme_light() + scale_x_continuous(expand = c(0,0)) + scale_y_continuous(expand = c(0,0)) +
    geom_contour(color = "black") + theme(text = element_text(size = 12), legend.position = "none") + 
    geom_text_contour(label.placement = label_placement_n(1),
                      rotate = FALSE, stroke = 0.1, skip = 0) + xlab(expression(x[5])) + ylab(expression(x[6])) + 
    scale_fill_divergent(low ="#67a9cf", mid = "#f7f7f7", high = "#ef8a62")  + labs(title = expression(h[2])) + 
    theme(plot.title = element_text(hjust = 0.5))+
    theme(plot.margin = unit(c(0.01,0.1,0.1,0.1), "cm"))) #t,r,b,l

(p4 <- ggplot(h2_pred, aes(x_5, x_6, z = h2)) + geom_contour_fill(aes(z = h2), breaks = br) + 
    theme_light() + scale_x_continuous(expand = c(0,0)) + scale_y_continuous(expand = c(0,0)) +
    geom_contour(color = "black") + theme(text = element_text(size = 12), legend.position = "none") + 
    geom_text_contour(label.placement = label_placement_n(1), 
                      rotate = FALSE, stroke = 0.1, skip = 0) + xlab(expression(x[5])) + ylab(expression(x[6])) + 
    scale_fill_divergent(low ="#67a9cf", mid = "#f7f7f7", high = "#ef8a62") + labs(title = expression(hat(h[2]))) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(plot.margin = unit(c(0.01,0.1,0.1,0.1), "cm"))) #t,r,b,l)


pp <- plot_grid(p1,p2,p3,p4, align = "h", nrow = 2)
ggsave(pp, filename = "Figure3_ECML.pdf", width = 10, height = 5, dpi = 300)

### Figure: supplement material ###

# 6 out of 10 DGPs were not shown in the main paper, only the 4 that can be compared to TBM
DGPs_main <- c("g[1] * eta[1] * sigma[1]","g[1] * eta[2] * sigma[1]","g[2] * eta[1] * sigma[1]","g[2] * eta[2] * sigma[1]")
h1_remain <- h1[!DGP %in% DGPs_main, ]
h2_remain <- h2[!DGP %in% DGPs_main, ]

ress <- rbind(h1_remain,h2_remain)
ress[ , TrafoFunc := factor(rep(c("h1","h2"), each = 480), labels = c(expression(h[1]),expression(h[2])))]
ress$inter <- interaction(ress$`BSP Order`, ress$`Sample size`)
levels(ress$inter) <- c("15/500","25/500", "15/3000","25/3000")
#levels(ress$DGP)[1:4] <- paste0("DGP~",1:4)

(gg <- ress %>% ggplot(aes(x = inter, y = log(`(RI)MSE`))) +
    geom_boxplot(position = "dodge") + scale_fill_jco() + facet_grid(DGP ~ TrafoFunc, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 10), legend.position = "none")  +
    xlab("Order Polynomials/Sample Size") + ylab("log((RI)MSE)") + labs(fill = ""))
ggsave(gg, filename = "res_ECML_supplement.pdf", width = 6, height = 6)