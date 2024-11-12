#args <- commandArgs(TRUE)

#i <- args[1]

library(tidyverse)
library(magrittr)
library(Matrix)
library(spiox)

RhpcBLASctl::blas_set_num_threads(2)
RhpcBLASctl::omp_set_num_threads(2)

# many outcomes, parsimonious matern model

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

q <- 24
g <- glue::glue


mcmc <- 10000
mcmc_keep <- 5000
nr <- 2500
nout <- 400

Sigbuild <- function(spiox_out){
  result <- 1:10000 %>% sapply(\(i) with(spiox_out, sqrt(diag(theta[2,,i])) %*% Sigma[,,i] %*% sqrt(diag(theta[2,,i]))  )) %>%
    array(dim=c(q,q,10000))
  return(
    result
  )
}

for(i in 1:20){
  cat(i, "\n")
  
  load(g("simulations/lmc_m/data_{i}.RData"))
  Y <- simdata %>% dplyr::select(contains("Y."))
  Y_out <- Y %>% tail(nout)
  Y_out_arr <- 1:mcmc %>% lapply(\(i) Y_out) %>% abind::abind(along=3)
  coords_all <- simdata %>% dplyr::select(contains("coords")) 
  cx <- coords_all %>% head(nr) %>% as.matrix()
  
  cat("load_results", "\n")
  
  load(g("simulations/lmc_m/spiox_metrop_{i}.RData"))
  time_metrop <- total_time
  load(g("simulations/lmc_m/spiox_gibbs_{i}.RData"))
  time_gibbs <- total_time
  load(g("simulations/lmc_m/spiox_clust_{i}.RData"))
  time_clust <- total_time
  load(g("simulations/lmc_m/meshed_{i}.RData"))
  
  meshed_outdata <- simdata %>% mutate(sample = c(rep("insample", nr), rep("outsample", nout))) %>% 
    left_join(spmeshed_out$coordsdata %>% mutate(ix=1:(nr+nout)), by=c("coords.Var1"="Var1", "coords.Var2"="Var2"))
  
  meshed_outsample <- meshed_outdata %>%
    dplyr::filter(sample=="outsample") %$% ix
  
  Y_target_meshed <- meshed_outdata %>% dplyr::select(contains("Y.")) %>% `[`(meshed_outsample,) 
  meshed_predicts <- list(Y = spmeshed_out$yhat_mcmc %>% abind::abind(along=3) %>% `[`(meshed_outsample,,))
  
  # meshed_time
  load(g("simulations/lmc_m/nngp_{i}.RData"))
  nngp_predicts <- list()
  nngp_predicts$Y <- array(0, dim=c(400, 24, 5000))
  for(j in 1:24){
    cat(j, " ")
    nngp_predicts$Y[,j,] <- nngp_results[[j]]$predicts$p.y.0
  }
  nngp_Sigma <- nngp_predicts$Y %>% apply(3, \(x) cor(x)) %>% array(dim=c(q,q,5000))
  nngp_Cor0 <- nngp_Sigma %>% apply(1:2, mean)
  cat("\n")
  # nngp_time
  
  
  cat("predictions", "\n")
  
  Y_out_gdiff <- rowSums(Y_out[,1:12]) - rowSums(Y_out[,13:24])
  target <- spiox_clust_predicts
  
  calc_performance <- function(target){
    perf1 <- 1:q %>% sapply(\(i) sqrt(mean((apply(target$Y[,i,], 1, mean) - Y_out[,i])^2)) )
    
    target_gdiff <- apply(target$Y[,1:12,], c(1,3), sum) - apply(target$Y[,13:24,], c(1,3), sum)
    perf2 <- sqrt( mean( ( apply(target_gdiff, 1, mean) - Y_out_gdiff)^2 ) )
    
    return(list(marginal_rmspe = perf1,
                gdiff_rmspe = perf2))
  }
  
  table_marginals <- \(listx){
    listx %>% lapply(\(x) x$marginal_rmspe) %>% do.call(cbind, .)
  }
  
  table_gdiff <- \(listx){
    listx %>% lapply(\(x) x$gdiff_rmspe) %>% do.call(cbind, .)
  }
  
  perf_spiox_gibbs <- calc_performance(spiox_gibbs_predicts)
  perf_spiox_metrop <- calc_performance(spiox_metrop_predicts)
  perf_spiox_clust <- calc_performance(spiox_clust_predicts)
  perf_spmeshed <- calc_performance(meshed_predicts)
  perf_nngp <- calc_performance(nngp_predicts)
  
  cat("cor at zero", "\n")
  
  spiox_cor_at_zero <- function(spiox_out){
    spiox_theta <- spiox_out$theta
    
    Sig <- Sigbuild(spiox_out)
    
    spiox_theta[2,,] <- 1
    spiox_scaling_factors <- scaling_factor_at_zero(cx, spiox_theta %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean))
    spiox_scaling_factors[upper.tri(spiox_scaling_factors)] <- spiox_scaling_factors[lower.tri(spiox_scaling_factors)]
    
    return(
      Sig %>% apply(3, \(s) cov2cor(s * spiox_scaling_factors)) %>% array(dim=c(q,q,10000)) )
  }
  
  
  theta_mat <- rbind(optlist, 1, 1, 1e-15)
  
  spiox_scaling_factors <- scaling_factor_at_zero(cx, theta_mat) 
  spiox_scaling_factors[upper.tri(spiox_scaling_factors)] <- spiox_scaling_factors[lower.tri(spiox_scaling_factors)]
  Cor_at_zero <- 
    cov2cor(Sigma * spiox_scaling_factors)
  
  spiox_gibbs_Cor0 <- spiox_cor_at_zero(spiox_gibbs_out) %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean)
  spiox_metrop_Cor0 <- spiox_cor_at_zero(spiox_metrop_out) %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean)
  spiox_clust_Cor0 <- spiox_cor_at_zero(spiox_clust_out) %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean)
  
  omega_spmeshed <- 1:nr %>% lapply(\(m) with(spmeshed_out, 
              cov2cor(tcrossprod(lambda_mcmc[,,m]) + diag(tausq_mcmc[,m])))) %>%
    abind::abind(along=3) %>% tail(c(NA, NA, 5000)) %>% apply(1:2, mean)
  
  
  
  results_gdiff <- table_gdiff(list(perf_spiox_gibbs, perf_spiox_metrop, perf_spiox_clust, perf_spmeshed, perf_nngp)) %>% 
    as.data.frame() %>% mutate(sim = i)
  results_margs <- table_marginals(list(perf_spiox_gibbs, perf_spiox_metrop, perf_spiox_clust, perf_spmeshed, perf_nngp)) %>% 
    as.data.frame() %>% mutate(sim = i, variable=1:n())
  colnames(results_gdiff)[1:5] <- colnames(results_margs)[1:5] <- c("spiox_gibbs", "spiox_metrop", "spiox_clust", "spmeshed", "nngp") 
  
  mat_lt <- function(mat){
    return( mat %>% `[`(lower.tri(.)) )
  }
  
  results_estcor <- data.frame(spiox_gibbs = mat_lt(spiox_gibbs_Cor0),
                               spiox_metrop = mat_lt(spiox_metrop_Cor0),
                               spiox_clust = mat_lt(spiox_clust_Cor0),
                               spmeshed = mat_lt(omega_spmeshed),
                               nngp = mat_lt(nngp_Cor0),
                               true = mat_lt(Cor_at_zero),
                               sim = i)
  
  
  timings <- list(sim=i, spiox_gibbs = time_gibbs["elapsed"],
                  spiox_metrop = time_metrop["elapsed"],
                  spiox_clust = time_clust["elapsed"],
                  meshed = meshed_time["elapsed"],
                  nngp = nngp_time["elapsed"])
  
  results <- list(sim=i, timings=timings,
                  gdiff=results_gdiff, 
                       margs=results_margs,
                       cor0 = results_estcor)
  
  
  save(file=g("simulations/lmc_m/results_{i}.RData"), list=c("results"))

}