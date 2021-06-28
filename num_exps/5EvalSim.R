rm(list = ls())
library(ggplot2)
library(ggsci)
library(magrittr)
library(data.table)

# set wd accordingly

# STM Boost results
load("h1_res_boost_ECML.RData")
load("h2_res_boost_ECML.RData")

# DCTM results
load("h1_res_dctm_ECML.RData")
load("h2_res_dctm_ECML.RData")

#### h1

# Subset specifications of DCTMs for the numerical experiments that are also handled by STM Boost
h1_net_res <- h1[DGP %in% c("g[1] * eta[1] * sigma[1]","g[1] * eta[2] * sigma[1]","g[2] * eta[1] * sigma[1]","g[2] * eta[2] * sigma[1]"), ]

h1_res <- rbind(h1_b_res, h1_net_res)
h1_res$Method <- rep(c("TBM Shift","DCTM"), each = 320)

(gg <- h1_res %>% ggplot(aes(x = `BSP Order`, y = log(`(RI)MSE`), fill = DGP)) +
    geom_boxplot(position="dodge", aes(fill = Method)) + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 16), legend.position = "bottom"))
# ggsave(gg, filename = "h1_res.pdf", width = 8, height = 10)

#### h2

# Subset specifications of DCTMs for the numerical experiments that are also handled by STM Boost
h2_net_res <- h2[DGP %in% c("g[1] * eta[1] * sigma[1]","g[1] * eta[2] * sigma[1]","g[2] * eta[1] * sigma[1]","g[2] * eta[2] * sigma[1]"), ]

h2_res <- rbind(h2_b_res, h2_net_res)
h2_res$Method <- rep(c("TBM Shift","DCTM"), each = 320)

(gg <- h2_res %>% ggplot(aes(x = `BSP Order`, y = log(`(RI)MSE`), fill = DGP)) +
    geom_boxplot(position = "dodge", aes(fill = Method)) + scale_fill_jco() + facet_grid(DGP ~ `Sample size`, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 16), legend.position = "top"))
# ggsave(gg, filename = "h2_res.pdf", width = 8, height = 10)

#### Merge h1 and h2

ress <- rbind(h1_res,h2_res)
ress[ , TrafoFunc := factor(rep(c("h1","h2"), each = 640), labels = c(expression(h[1]),expression(h[2])))]
ress$inter <- interaction(ress$`BSP Order`, ress$`Sample size`)
levels(ress$inter) <- c("15/500","25/500", "15/3000","25/3000")
levels(ress$DGP)[1:4] <- paste0("DGP~",1:4)

(gg <- ress %>% ggplot(aes(x = inter, y = log(`(RI)MSE`), fill = DGP)) +
    geom_boxplot(position = "dodge", aes(fill = Method)) + scale_fill_jco() + facet_grid(DGP ~ TrafoFunc, scales = "free", labeller = label_parsed) +
    theme_bw() + theme(text = element_text(size = 16), legend.position = "none") +
    xlab("Order Polynomials/Sample Size") + ylab("log((RI)MSE)") + labs(fill = ""))
ggsave(gg, filename = "Figure_1_ECML.pdf", width = 10, height = 6, dpi = 300)

#### Summary of results

# median RIMSE of reps
ress[, med_mse := median(`(RI)MSE`), by = list(`DGP`,`inter`,`Method`,`TrafoFunc`)]
sum_mary <- ress[seq(1,1261,20),]
sum_mary <- sum_mary[order(inter, `DGP`,`TrafoFunc`,`med_mse`),]

table(sum_mary[TrafoFunc == "h[1]", Method][seq(1,31,2)])/16
table(sum_mary[TrafoFunc == "h[2]", Method][seq(1,31,2)])/16

# mean RIMSE of reps
ress[, mean_mse := mean(`(RI)MSE`), by = list(`DGP`,`inter`,`Method`,`TrafoFunc`)]
sum_mary <- ress[seq(1,1261,20),]
sum_mary <- sum_mary[order(inter, `DGP`,`TrafoFunc`,`mean_mse`),]

table(sum_mary[TrafoFunc == "h[1]", Method][seq(1,31,2)])/16
table(sum_mary[TrafoFunc == "h[2]", Method][seq(1,31,2)])/16
