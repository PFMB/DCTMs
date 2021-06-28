rm(list = ls())
library(ggplot2)
library(ggsci) # journal style plot colors
library(data.table)
library(tidyverse)

# set wd accordingly

reps <- 20 # set number of repetitions

# measure for evaluation
relimse <- function(prd, trth) sum(c(((prd - trth)^2)), na.rm = T) / sum(c(trth^2), na.rm = T)
mse <- function(prd, trth) sum((prd - trth)^2)/length(prd)

# load results of numerical experiments
boos_res <- list.files(pattern = "^boost_res_n_")
boos_res <- sapply(boos_res, load, envir = .GlobalEnv)

### DGPs handled by STM Boosting: "y_g1_e1_s1", "y_g1_e2_s1", "y_g2_e1_s1", "y_g2_e2_s1"

# flipped sign compared to DGP, due to specification of boosting
true_coef <- c("x1" = -1)

g1 <- function(x) sinh(x)
g2 <- function(x) {
  y <- numeric(length(x))
  y[0 < x & x <= 1] <- log(x[0 < x & x <= 1]) + 1
  y[1 < x & x < 2] <- x[1 < x & x < 2]
  y[x >= 2] <- 1 + exp(x[x >= 2] - 2)
  y
}

b_res <- lapply(boos_res, function(d_set) {
  
  #### g_1, eta_1, sigma_1
  g1_e1_s1 <- lapply(eval(parse(text = d_set)), `[[`, "y_g1_e1_s1")
  
  # h1
  pr_h1_g1_e1_s1 <- lapply(g1_e1_s1, `[[`, "h1_hat")
  tr_h1_g1_e1_s1 <- lapply(g1_e1_s1, function(b) g1(b$y_gr[[1]]))
  h1_g1_e1_s1 <- sapply(1:reps, function(idx) relimse(pr_h1_g1_e1_s1[[idx]], tr_h1_g1_e1_s1[[idx]]))
  # plot(pr_h1_g1_e1_s1[[1]], ylim = c(-5,5), col = "red")
  # par(new = TRUE)
  # plot(tr_h1_g1_e1_s1[[1]], ylim = c(-5,5))
  
  # h2
  h2_g1_e1_s1 <- lapply(g1_e1_s1, `[[`, "bbs_pred")
  tr_h2_g1_e1_s1 <- sapply(h2_g1_e1_s1, function(d) sin(d$x))
  pr_h2_g1_e1_s1 <- -1*sapply(h2_g1_e1_s1, `[[`, "prd")
  # plot(h2_g1_e1_s1[[1]]$x, pr_h2_g1_e1_s1[,1], ylim = c(-1,1), col = "red")
  # par(new = TRUE)
  # plot(h2_g1_e1_s1[[1]]$x, tr_h2_g1_e1_s1[,1], ylim = c(-1,1))
  h2_g1_e1_s1 <- sapply(1:reps, function(c_idx) relimse(pr_h2_g1_e1_s1[,c_idx], tr_h2_g1_e1_s1[,c_idx]))
  
  #### g_2, eta_1, sigma_1
  g2_e1_s1 <- lapply(eval(parse(text = d_set)), `[[`, "y_g2_e1_s1")
  
  # h1
  pr_h1_g2_e1_s1 <- lapply(g2_e1_s1, `[[`, "h1_hat")
  tr_h1_g2_e1_s1 <- lapply(g2_e1_s1, function(b) g2(b$y_gr[[1]]))
  h1_g2_e1_s1 <- sapply(1:reps, function(idx) relimse(pr_h1_g2_e1_s1[[idx]], tr_h1_g2_e1_s1[[idx]]))
  # plot(pr_h1_g2_e1_s1[[1]], ylim = c(-5,5), col = "red")
  # par(new = TRUE)
  # plot(tr_h1_g2_e1_s1[[1]], ylim = c(-5,5))
  
  # h2
  h2_g2_e1_s1 <- lapply(g2_e1_s1, `[[`, "bbs_pred")
  tr_h2_g2_e1_s1 <- sapply(h2_g2_e1_s1, function(d) sin(d$x))
  pr_h2_g2_e1_s1 <- -1*sapply(h2_g2_e1_s1, `[[`, "prd")
  # plot(h2_g2_e1_s1[[1]]$x, pr_h2_g2_e1_s1[,1], ylim = c(-1,1), col = "red")
  # par(new = TRUE)
  # plot(h2_g2_e1_s1[[1]]$x, tr_h2_g2_e1_s1[,1], ylim = c(-1,1))
  h2_g2_e1_s1 <- sapply(1:reps, function(c_idx) relimse(pr_h2_g2_e1_s1[,c_idx], tr_h2_g2_e1_s1[,c_idx]))
  
  #### g_1, eta_2, sigma_1
  g1_e2_s1 <- lapply(eval(parse(text = d_set)), `[[`, "y_g1_e2_s1")
  
  # h1
  pr_h1_g1_e2_s1 <- lapply(g1_e2_s1, `[[`, "h1_hat")
  tr_h1_g1_e2_s1 <- lapply(g1_e2_s1, function(b) g1(b$y_gr[[1]]))
  h1_g1_e2_s1 <- sapply(1:reps, function(idx) relimse(pr_h1_g1_e2_s1[[idx]], tr_h1_g1_e2_s1[[idx]]))
  # plot(pr_h1_g1_e2_s1[[1]], ylim = c(-5,5), col = "red")
  # par(new = TRUE)
  # plot(tr_h1_g1_e2_s1[[1]], ylim = c(-5,5))
  
  # h2 sin(x5)
  h2_g1_e2_s1 <- lapply(g1_e2_s1, `[[`, "bbs_pred")
  h2_gr <- lapply(h2_g1_e2_s1,  `[[`, "x")
  tr_h2_g1_e2_s1 <- sapply(h2_gr, function(x5) sin(x5))
  pr_g1_e2_s1 <- -1*sapply(h2_g1_e2_s1, `[[`, "prd")
  # plot(h2_g1_e2_s1[[1]]$x, pr_g1_e2_s1[,1], ylim = c(-1,1), col = "red")
  # par(new = TRUE)
  # plot(h2_g1_e2_s1[[1]]$x, tr_h2_g1_e2_s1[,1], ylim = c(-1,1))
  h2_g1_e2_s1 <- sapply(1:reps, function(c_idx) relimse(pr_g1_e2_s1[,c_idx], tr_h2_g1_e2_s1[,c_idx]))
  
  # beta
  beta <- -1*unlist(sapply(g1_e2_s1, `[[`, "bols_coef"))
  mse_beta <- mean((beta - 1)^2)
  h2_g1_e2_s1 <- h2_g1_e2_s1 + mse_beta 
  
  #### g_2, eta_2, sigma_1
  g2_e2_s1 <- lapply(eval(parse(text = d_set)), `[[`, "y_g2_e2_s1")
  
  # h1
  pr_h1_g2_e2_s1 <- lapply(g2_e2_s1, `[[`, "h1_hat")
  tr_h1_g2_e2_s1 <- lapply(g2_e2_s1, function(b) g2(b$y_gr[[1]]))
  h1_g2_e2_s1 <- sapply(1:reps, function(idx) relimse(pr_h1_g2_e2_s1[[idx]], tr_h1_g2_e2_s1[[idx]]))
  # plot(pr_h1_g2_e2_s1[[1]], ylim = c(-5,5), col = "red")
  # par(new = TRUE)
  # plot(pr_h1_g2_e2_s1[[1]], ylim = c(-5,5))
  
  # h2
  h2_g2_e2_s1 <- lapply(g2_e2_s1, `[[`, "bbs_pred")
  h2_gr <- lapply(h2_g2_e2_s1,  `[[`, "x")
  tr_g2_e2_s1 <- sapply(h2_gr, function(x5) sin(x5))
  pr_g2_e2_s1 <- -1*sapply(h2_g2_e2_s1, `[[`, "prd")
  # plot(h2_gr[[2]], pr_g2_e2_s1[,1], ylim = c(-1,1), col = "red")
  # par(new = TRUE)
  # plot(h2_gr[[2]], tr_g2_e2_s1[,1], ylim = c(-1,1))
  h2_g2_e2_s1 <- sapply(1:reps, function(c_idx) relimse(pr_g2_e2_s1[,c_idx], tr_g2_e2_s1[,c_idx]))
  
  # beta
  beta <- -1*unlist(sapply(g2_e2_s1, `[[`, "bols_coef"))
  mse_beta <- mean((beta - 1)^2)
  h2_g2_e2_s1 <- h2_g2_e2_s1 + mse_beta 
  
  res_h1 <- data.table(cbind(h1_g1_e1_s1, h1_g2_e1_s1, h1_g1_e2_s1, h1_g2_e2_s1))
  colnames(res_h1) <- c("h1_g1_e1_s1", "h1_g2_e1_s1", "h1_g1_e2_s1", "h1_g2_e2_s1")
  
  res_h2 <- data.table(cbind(h2_g1_e1_s1, h2_g2_e1_s1, h2_g1_e2_s1, h2_g2_e2_s1))
  colnames(res_h2) <- c("h2_g1_e1_s1", "h2_g2_e1_s1", "h2_g1_e2_s1", "h2_g2_e2_s1")
  list(res_h1 = res_h1, res_h2 = res_h2)
})

### h1

h1_b_res <- lapply(b_res, `[[`, "res_h1") 
sett <- cbind(rep(c("M = 15","M = 25"), 2), # needs to match order of boos_res
              c(rep("n = 3000", 2), rep("n = 500", 2)))
for (idx in 1:length(h1_b_res)) {
  d <- h1_b_res[[idx]]
  d1 <- data.table(matrix(rep(sett[idx,], reps), ncol = 2, byrow = TRUE))
  setnames(d1, c("V1","V2"), c("BSP Order", "Sample size"))
  h1_b_res[[idx]] <- cbind(d, d1)
}

h1_b_res <- rbindlist(h1_b_res)
h1_b_res <- melt(h1_b_res, id.vars = c("BSP Order","Sample size"), variable.name = "DGP", value.name = "(RI)MSE")
h1_b_res[, `BSP Order` := factor(`BSP Order`)]
h1_b_res[, `Sample size` := factor(`Sample size`, levels = c("n = 500", "n = 3000"))]

# labeller = label_parsed changes column as well as row labels, even if we only want to change the rows
h1_b_res[, `Sample size` := factor(`Sample size`, labels = c(expression(n == 500), expression(n == 3000)))]

lbl <- c(expression(g[1] * eta[1] * sigma[1]),
         expression(g[2] * eta[1] * sigma[1]),
         expression(g[1] * eta[2] * sigma[1]),
         expression(g[2] * eta[2] * sigma[1]))
h1_b_res[, DGP := factor(DGP, labels = lbl)]
save(h1_b_res, file = "h1_res_boost_ECML.RData")

(gg <- h1_b_res %>% ggplot(aes(x = `BSP Order`, y = `(RI)MSE`, fill = DGP)) +
    geom_boxplot() + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 14), legend.position = "none"))

### h2

h2_b_res <- lapply(b_res, `[[`, "res_h2") 

for (idx in 1:length(h2_b_res)) {
  d <- h2_b_res[[idx]]
  d1 <- data.table(matrix(rep(sett[idx,], reps), ncol = 2, byrow = TRUE))
  setnames(d1, c("V1","V2"), c("BSP Order", "Sample size"))
  h2_b_res[[idx]] <- cbind(d, d1)
}

h2_b_res <- rbindlist(h2_b_res)
h2_b_res <- melt(h2_b_res, id.vars = c("BSP Order","Sample size"), variable.name = "DGP", value.name = "(RI)MSE")
h2_b_res[, `BSP Order` := factor(`BSP Order`)]
h2_b_res[, `Sample size` := factor(`Sample size`, levels = c("n = 500", "n = 3000"))]

# labeller = label_parsed changes column as well as row labels, even if we only want to change the rows
h2_b_res[, `Sample size` := factor(`Sample size`, labels = c(expression(n == 500), expression(n == 3000)))]

h2_b_res[, DGP := factor(DGP, labels = lbl)]
save(h2_b_res, file = "h2_res_boost_ECML.RData")

# (gg <- h2_b_res %>% ggplot(aes(x = `BSP Order`, y = `(RI)MSE`, fill = DGP)) +
#     geom_boxplot() + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
#     theme_bw() + theme(text = element_text(size = 14), legend.position = "none"))
