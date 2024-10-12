rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(spiox)

g <- glue::glue

s <- 2

filelist <- Sys.glob(g("simulations/spiox/*{s}*"))

nngp_predicts <- array(0, dim=c(n_out, 2, 2500))
nngp_predicts[,1,] <- nngp_pred.1$p.y.0
nngp_predicts[,2,] <- nngp_pred.2$p.y.0

uni_perf <- function(arr, Y){
  1:2 %>% sapply(\(j) arr[,j,] %>% apply(1, mean) %>% cbind(Y[,j]) %>% cor() %>% `[`(1,2) )
}


filelist %>% sapply(\(f) load(f, envir=globalenv()))

n_in <- 2500
n_out <- 400



Yi <- simdata %>% dplyr::select(contains("Y.")) %>% head(n_in)
quants <- Yi %>% apply(2, quantile, .5)
Yo <- simdata %>% dplyr::select(contains("Y.")) %>% tail(n_out)
Y_over <- Yo %>% apply(1, \(y) y>quants) %>% t() 
Y_both_over <- Y_over %>% apply(1, \(yo) prod(yo))
Yo_product <- Yo[,1]*Yo[,2]

# spiox response
spiox_predicts$Y %>% apply(c(1,3), \(y) prod(y>quants)) %>% mean
Y_sr_prod <- spiox_predicts$Y %>% apply(c(1,3), \(y) prod(y))
Y_sr_prod %>% apply(1, mean) %>% cbind(Yo_product) %>% cor() %>% `[`(1,2)
uni_perf(spiox_predicts$Y, Yo)

# spiox latent
spiox_latent_predicts$Y %>% apply(c(1,3), \(y) prod(y>quants)) %>% mean
Y_sl_prod <- spiox_latent_predicts$Y %>% apply(c(1,3), \(y) prod(y))
Y_sl_prod %>% apply(1, mean) %>% cbind(Yo_product) %>% cor() %>% `[`(1,2)
uni_perf(spiox_latent_predicts$Y, Yo)

# nngp univariate
nngp_predicts %>% apply(c(1,3), \(y) prod(y>quants)) %>% mean
Y_nn_prod <- nngp_predicts %>% apply(c(1,3), \(y) prod(y))
Y_nn_prod %>% apply(1, mean) %>% cbind(Yo_product) %>% cor() %>% `[`(1,2)
uni_perf(nngp_predicts, Yo)


# marg intervals
lowy <- spiox_latent_predicts$Y[,1,] %>% apply(1, quantile, .025)
highy <- spiox_latent_predicts$Y[,1,] %>% apply(1, quantile, .975)
mean( (lowy < Yo[,1]) * (Yo[,1] < highy) )

# joint uncertainty bands
spiox_cloud <- 2401:2500 %>% lapply(\(i) data.frame(Y=spiox_predicts$Y[,,i] - Yo
                                                      , i=i)) %>% bind_rows()
spiox_predicts$Y %>% apply(3, \(x) cor(x)[2,1]) %>% mean()

spiox_cloud2 <- 2401:2500 %>% lapply(\(i) data.frame(Y=spiox_latent_predicts$Y[,,i] - Yo
                                                     , i=i)) %>% bind_rows()
spiox_latent_predicts$Y %>% apply(3, \(x) cor(x)[2,1]) %>% mean()

nngp_cloud <- 2401:2500 %>% lapply(\(i) data.frame(Y=nngp_predicts[,,i] - Yo
                                                   , i=i)) %>% bind_rows()
nngp_predicts %>% apply(3, \(x) cor(x)[2,1]) %>% mean()

colnames(spiox_cloud) <- colnames(spiox_cloud2) <- colnames(nngp_cloud) <-
  c("Y.1", "Y.2", "i")

(nngp_plot <- nngp_cloud %>% 
    ggplot(aes(x=Y.1, y=Y.2)) +
    geom_density_2d_filled() + 
    scale_fill_scico_d(palette="grayC", direction=-1) +
    theme_minimal() + theme(legend.position="none") +
    labs(x="Outcome 1", y=NULL) +
    ggtitle("Univariate NNGP") + 
    coord_cartesian(xlim=c(-1.5, 1.5), ylim=c(-1.5, 1.5))
  )

(spiox_plot <- spiox_cloud %>% 
  ggplot(aes(x=Y.1, y=Y.2)) +
  geom_density_2d_filled() + 
    scale_fill_scico_d(palette="grayC", direction=-1) +
    theme_minimal() + theme(legend.position="none") +
    labs(x="Outcome 1", y="Outcome 2") +
    ggtitle("IOX Response") + 
    coord_cartesian(xlim=c(-1.5, 1.5), ylim=c(-1.5, 1.5)))

(spiox_latent_plot <- spiox_cloud2 %>% 
    ggplot(aes(x=Y.1, y=Y.2)) +
    geom_density_2d_filled() + 
    scale_fill_scico_d(palette="grayC", direction=-1) +
    theme_minimal() + theme(legend.position="none") +
    labs(x="Outcome 1", y=NULL) +
    ggtitle("IOX Latent") + 
    coord_cartesian(xlim=c(-1.5, 1.5), ylim=c(-1.5, 1.5)))

gridExtra::grid.arrange(spiox_plot, spiox_latent_plot, nngp_plot, nrow=1)








# 
g <- glue::glue
dgm <- "matern2"
coveragef <- function(Yarr, Yo){
  covs <- rep(1, ncol(Yarr))
  for(j in 1:ncol(Yarr)){
    Yarr[,j,] %>% apply(1, quantile, .025) -> lowy
    Yarr[,j,] %>% apply(1, quantile, .975) -> highy
    covs[j] <- mean( (lowy < Yo[,j])*(Yo[,j] < highy) )
  }
  return(covs)
}
nsim <- length(Sys.glob("~/spiox/simulations/matern2/data*"))
all_covs <- matrix(0, nrow=nsim, ncol=2)
for(i in 1:nsim){
  cat(i, "\n")
  load(g("~/spiox/simulations/{dgm}/data_{i}.RData"))
  load(g("~/spiox/simulations/{dgm}/spiox_{i}.RData"))
  
  Y <- simdata %>% dplyr::select(contains("Y."))
  Yo <- Y %>% tail(400)
  all_covs[i, ] <- coveragef(spiox_predicts$Y, Yo)
}

