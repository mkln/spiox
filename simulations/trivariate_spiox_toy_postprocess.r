rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(spiox)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

rtail <- \(x, nt){
  tail(x, nt)
}
tailh <- \(x, nt){
  tail(x, round(length(x)/2))
}
rtailh <- \(x){ rtail(x, round(nrow(x)/2)) }
ctail <- \(x, nt){
  tail(x, c(NA,nt))
}
ctailh <- \(x){ ctail(x, round(ncol(x)/2)) }
stail <- \(x, nt){
  tail(x, c(NA,NA,nt))
}
stailh <- \(x){ stail(x, round(dim(x)[3]/2)) }

perturb <- function(x, sd=1){
  return(x + matrix(rnorm(prod(dim(x)), sd), ncol=ncol(x)))
}

set.seed(1)


q <- 3

theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(phi <- 20, phi, phi)
theta_mat[2,] <- c(1, 1, 1)
theta_mat[3,] <- c(nu1 <- 0.5, nu2 <- .8, nu3 <- 1.2)
theta_mat[4,] <- rep(1e-13, q)

sds <- c(1, 1, 1)
Omega <- matrix(c(1,-.9,0.7,-.9,1,-.5,0.7,-.5,1), ncol=3)
Sigma <- diag(sds) %*% Omega %*% diag(sds)

St <- chol(Sigma)
S <- t(St)

# spatial
cx_in <- matrix(runif(2500*2), ncol=2) #2500
colnames(cx_in) <- c("Var1","Var2")
n_in <- nrow(cx_in)
which_in <- 1:n_in

xout <- seq(0, 1, length.out=50) #50
coords_out <- expand.grid(xout, xout)
cx_out <- as.matrix(coords_out)
n_out <- nrow(cx_out)
which_out <- (n_in+1):(n_in+n_out)

cx_all <- rbind(cx_in, cx_out)
nr_all <- nrow(cx_all)


cx_12 <- rbind(cbind(cx_all, 1), cbind(cx_all, 2), cbind(cx_all, 3))

V <- Omega #cbind(c(1, rhocorr), c(rhocorr, 1)) 
St <- chol(V)

phi <- 30
sigma11 <- sigma22 <- sigma33 <- 1
matern_scaling_factors <- diag(q)
matern_scaling_factors[1,2] <- matern_scaling_factors[2,1] <-
  sqrt(sigma11 * sigma22) * 
  sqrt(gamma(nu1+1))/sqrt(gamma(nu1)) * 
  sqrt(gamma(nu2+1))/sqrt(gamma(nu2)) * gamma(nu1/2 + nu2/2)/gamma(nu1/2 + nu2/2 + 1)
matern_scaling_factors[1,3] <- matern_scaling_factors[3,1] <-
  sqrt(sigma11 * sigma33) * 
  sqrt(gamma(nu1+1))/sqrt(gamma(nu1)) * 
  sqrt(gamma(nu3+1))/sqrt(gamma(nu3)) * gamma(nu1/2 + nu3/2)/gamma(nu1/2 + nu3/2 + 1)
matern_scaling_factors[2,3] <- matern_scaling_factors[3,2] <-
  sqrt(sigma22 * sigma33) * 
  sqrt(gamma(nu2+1))/sqrt(gamma(nu2)) * 
  sqrt(gamma(nu3+1))/sqrt(gamma(nu3)) * gamma(nu2/2 + nu3/2)/gamma(nu2/2 + nu3/2 + 1)

sigma12 <- V[2,1] * matern_scaling_factors[1,2]
sigma13 <- V[3,1] * matern_scaling_factors[1,3]
sigma23 <- V[3,2] * matern_scaling_factors[2,3]

if(F){
  # parsimonious multi matern
  Cparmat <- GpGpm::matern_multi(c(sigma11, sigma12, sigma22, sigma13, sigma23, sigma33, 
                                   1/phi, 1/phi, 1/phi, 1/phi, 1/phi, 1/phi, 
                                   nu1, nu1/2+nu2/2, nu2, nu1/2+nu3/2, nu2/2+nu3/2, nu3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3), cx_12)
  
  Lparmat <- t(chol(Cparmat))
  
  wvec <- Lparmat %*% rnorm(nr_all*q)
  W <- matrix(wvec, ncol=q)
  
  # matern
  Y_sp <- W 
  
  # regression
  p <- 1
  X <- matrix(1, ncol=1, nrow=nr_all) #%>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))
  
  Beta <- matrix(rnorm(q * p), ncol=q)
  
  Y_regression <- X %*% Beta
  #Error <- matrix(rnorm(nrow(Y_regression) * q),ncol=q) %*% diag(D <- runif(q, 0, 0.1))
  Y <- as.matrix(Y_sp + Y_regression) # + Error
  
  Y_in <- Y[which_in,]
  X_in <- X[which_in,,drop=F]
  
  if(F){
    df <- data.frame(cx_out, y=as.matrix(Y_sp[which_out,])) %>% 
      pivot_longer(cols=-c(Var1, Var2))
    ggplot(df, 
           aes(Var1, Var2, fill=value)) +
      geom_raster() +
      scale_fill_viridis_c() +
      facet_wrap(~name, ncol=5)
  }
  
  simdata <- data.frame(coords=cx_all, Y_spatial=Y_sp, Y=Y, X=X)
  #ggplot(simdata, aes(coords.Var1, coords.Var2, color=Y_spatial.1)) + geom_point() + scale_color_viridis_c()
  
  save(file=glue::glue("simulations/trivariate_matern_toy/data.RData"), 
       list=c("simdata", 
              "Beta", "D", "Y_in", "X_in", "which_in", "which_out",
              "Y_regression", #"Error", 
              "Y", "X", "W", "Lparmat",
              "nu1", "nu2", "nu3", "phi", "Sigma"))
} else {
  load("simulations/trivariate_matern_toy/data.RData")
}
##############################

simdata %>% tail(2500) %>% 
  dplyr::select(contains("coords"), contains("Y_")) %>% 
  pivot_longer(cols=-c(coords.Var1, coords.Var2)) %>% 
  ggplot(aes(coords.Var1, coords.Var2, fill=value)) + 
  geom_raster() + 
  scale_fill_viridis_c() + 
  facet_grid(~name) + 
  theme_minimal()



set.seed(1)

#theta_opts <- cbind(c(10, 1, .5, 1e-19), c(20, 1, 1, 1e-19))
theta_opts <- cbind(c(10, 2, nu1, 1e-1), c(20, 1, nu2, 1e-2), c(20, 1, 0.5, 1e-2))
##############################################

m_nn <- 20
mcmc <- 20000
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

# "spiox_out", "spiox_predicts", "estim_time", "predict_time", "total_time"
load("simulations/trivariate_matern_toy/spiox.RData")

# "spiox_latentq_out", "spiox_latentq_predicts", "latentq_estim_time", "latentq_predict_time"
load("simulations/trivariate_matern_toy/spiox_latentq.RData")

# "spiox_latentn_out", "spiox_latentn_predicts", "latentn_estim_time", "latentn_predict_time"
load("simulations/trivariate_matern_toy/spiox_latentn.RData")

# "spmeshed_out", "Ymesh_out", "meshed_time"
load("simulations/trivariate_matern_toy/meshed.RData")

# "fit2", "gpgpm"
load("simulations/trivariate_matern_toy/gpgpm.RData")

spiox_dens_plotter <- function(spiox_out, which_var, bw=.6){
  
  targets <- spiox_out$theta %>% stailh() %>% `[`(which_var,,) %>% t() %>% as.data.frame() %>% 
    mutate(m = 1:n())
  colnames(targets) <- paste0("targets_", colnames(targets))
  targets %<>% pivot_longer(cols=-targets_m)
  
  ggplot(targets, aes(value)) +
    geom_boxplot()+ #bw=bw) +
    facet_grid( ~ name) + theme_minimal()
  
  
}

spiox_cov_at_zero <- function(spiox_out){
  spiox_theta <- spiox_out$theta
  spiox_theta[2,,] <- 1
  spiox_scaling_factors <- scaling_factor_at_zero(cx_in, spiox_theta %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean))
  spiox_scaling_factors[upper.tri(spiox_scaling_factors)] <- spiox_scaling_factors[lower.tri(spiox_scaling_factors)]
  
  return(
    spiox_out$Sigma %>% apply(3, \(s) cov2cor(s * spiox_scaling_factors)) %>% array(dim=c(q,q,mcmc)) )
}

# cov at zero
V * matern_scaling_factors # true
spiox_out %>% spiox_cov_at_zero() %>% apply(1:2, mean)
spiox_latentq_out %>% spiox_cov_at_zero() %>% apply(1:2, mean)
spiox_latentn_out %>% spiox_cov_at_zero() %>% apply(1:2, mean)
spmeshed_out$lambda_mcmc %>% meshed:::cube_correl_from_lambda() %>% apply(1:2, mean)









