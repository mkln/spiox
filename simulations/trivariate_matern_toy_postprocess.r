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

nu1 <- 0.5
nu2 <- .8
nu3 <- 1.2

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



set.seed(1)

#theta_opts <- cbind(c(10, 1, .5, 1e-19), c(20, 1, 1, 1e-19))
theta_opts <- cbind(c(10, 2, nu1, 1e-1), c(20, 1, nu2, 1e-2), c(20, 1, 0.5, 1e-2))
##############################################

m_nn <- 20
mcmc <- 10000
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sigbuild <- function(spiox_out){
  return(
    1:mcmc %>% sapply(\(i) with(spiox_out, sqrt(diag(theta[2,,i])) %*% Sigma[,,i] %*% sqrt(diag(theta[2,,i]))  )) %>%
      array(dim=c(q,q,mcmc))
  )
}

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
  
  Sigbuild <- function(spiox_out){
    return(
      1:mcmc %>% sapply(\(i) with(spiox_out, sqrt(diag(theta[2,,i])) %*% Sigma[,,i] %*% sqrt(diag(theta[2,,i]))  )) %>%
        array(dim=c(q,q,mcmc))
    )
  }
  Sig <- Sigbuild(spiox_out)
  
  spiox_theta[2,,] <- 1
  spiox_scaling_factors <- scaling_factor_at_zero(cx_in, spiox_theta %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean))
  spiox_scaling_factors[upper.tri(spiox_scaling_factors)] <- spiox_scaling_factors[lower.tri(spiox_scaling_factors)]

  return(
    Sig %>% apply(3, \(s) cov2cor(s * spiox_scaling_factors)) %>% array(dim=c(q,q,mcmc)) )
}

# cov at zero
SS <- V * matern_scaling_factors # true


est_results <- list()

for( rr in 1:60 ){
  cat(rr, "\n")
  
  # "spiox_out", "spiox_predicts", "estim_time", "predict_time", "total_time"
  load(glue::glue("simulations/trivariate_matern_toy_reps/spiox_{rr}.RData"))
  
  # "spiox_latentq_out", "spiox_latentq_predicts", "latentq_estim_time", "latentq_predict_time"
  load(glue::glue("simulations/trivariate_matern_toy_reps/spiox_latentq_{rr}.RData"))
  
  # "spiox_latentn_out", "spiox_latentn_predicts", "latentn_estim_time", "latentn_predict_time"
  load(glue::glue("simulations/trivariate_matern_toy_reps/spiox_latentn_{rr}.RData"))
  
  # "spmeshed_out", "Ymesh_out", "meshed_time"
  load(glue::glue("simulations/trivariate_matern_toy_reps/meshed_{rr}.RData"))
  
  # "fit2", "gpgpm"
  load(glue::glue("simulations/trivariate_matern_toy_reps/gpgpm_{rr}.RData"))
  
  est_phi <- list( spiox_resp = spiox_out$theta %>% stail(5000) %>% `[`(1,,) %>% apply(1, mean),
                   spiox_latn = spiox_latentn_out$theta %>% stail(5000) %>% `[`(1,,) %>% apply(1, mean),
                   spiox_latq = spiox_latentq_out$theta %>% stail(5000) %>% `[`(1,,) %>% apply(1, mean),
                   meshed = spmeshed_out$theta_mcmc %>% `[`(1,,) %>% apply(1, mean),
                   gpgpm = diag(gpgpm$phi) )
  
  est_nu <- list( spiox_resp = spiox_out$theta %>% stail(5000) %>% `[`(3,,) %>% apply(1, mean),
                  spiox_latn = spiox_latentn_out$theta %>% stail(5000) %>% `[`(3,,) %>% apply(1, mean),
                  spiox_latq = spiox_latentq_out$theta %>% stail(5000) %>% `[`(3,,) %>% apply(1, mean),
                  meshed = spmeshed_out$theta_mcmc %>% `[`(2,,) %>% apply(1, mean),
                  gpgpm = diag(gpgpm$nu) )
  
  
  est_tsq <- list( spiox_resp = spiox_out$theta %>% stail(5000) %>% `[`(4,,) %>% apply(1, mean),
                   spiox_latn = spiox_latentn_out$Ddiag %>% ctail(5000) %>% apply(1, mean),
                   spiox_latq = spiox_latentq_out$Ddiag %>% ctail(5000) %>% apply(1, mean),
                   meshed = spmeshed_out$tausq_mcmc %>% apply(1, mean),
                   gpgpm = diag(gpgpm$tausq) )
  
  spiox_cor_at_zero <- function(spiox_out){
    spiox_theta <- spiox_out$theta
    
    Sigbuild <- function(spiox_out, mcmc=10000){
      return(
        1:mcmc %>% sapply(\(i) with(spiox_out, sqrt(diag(theta[2,,i])) %*% Sigma[,,i] %*% sqrt(diag(theta[2,,i]))  )) %>%
          array(dim=c(q,q,mcmc))
      )
    }
    Sig <- Sigbuild(spiox_out)
    
    spiox_theta[2,,] <- 1
    spiox_scaling_factors <- scaling_factor_at_zero(cx_in, spiox_theta %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean))
    spiox_scaling_factors[upper.tri(spiox_scaling_factors)] <- spiox_scaling_factors[lower.tri(spiox_scaling_factors)]
    
    return(
      Sig %>% apply(3, \(s) cov2cor(s * spiox_scaling_factors)) %>% array(dim=c(q,q,mcmc)) )
  }
  
  # covariance at zero: estimated
  spiox_SS_resp <- spiox_out %>% spiox_cor_at_zero() %>% stail(5000) %>% apply(1:2, mean)
  spiox_SS_latq <- spiox_latentq_out %>% spiox_cor_at_zero() %>% stail(5000) %>% apply(1:2, mean)
  spiox_SS_latn <- spiox_latentn_out %>% spiox_cor_at_zero() %>% stail(5000) %>% apply(1:2, mean)
  
  meshed_SS <- spmeshed_out$lambda_mcmc %>% meshed:::cube_correl_from_lambda() %>% stail(5000) %>% apply(1:2, mean)
  gpgpm_SS <- cov2cor(gpgpm$Sigma)
  
  
  est_Corr0 <- list( spiox_resp = spiox_SS_resp[lower.tri(spiox_SS_resp)],
                     spiox_latn = spiox_SS_latn[lower.tri(spiox_SS_latn)],
                     spiox_latq = spiox_SS_latq[lower.tri(spiox_SS_latq)],
                     meshed = meshed_SS[lower.tri(meshed_SS)], 
                     gpgpm = gpgpm_SS[lower.tri(gpgpm_SS)] ) %>% as.data.frame() %>% 
    mutate(parameter = c("corr21", "corr31", "corr32"))
  
  
  est_results[[rr]] <- bind_rows(
    as.data.frame(est_phi) %>% mutate(parameter = paste0("phi", 1:n())),
    as.data.frame(est_nu) %>% mutate(parameter = paste0("nu", 1:n())),
    as.data.frame(est_tsq) %>% mutate(parameter = paste0("tausq", 1:n())),
    est_Corr0
  ) %>% mutate(rep = rr) %>% pivot_longer(cols=-c(parameter, rep), names_to = "model")
  
  
}


save(file="simulations/trivariate_matern_toy_reps/estimation_results_all.RData", list="est_results")

est_results %>% bind_rows() %>% group_by(model, parameter) %>% summarise_all(mean) %>% pivot_wider(id_cols=model, names_from = parameter)



