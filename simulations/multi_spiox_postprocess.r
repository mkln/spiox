rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(spiox)

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
i <- 2

results <- list()

for(i in 1:20){
  cat(i, "\n")
  
  load(g("simulations/spiox_m/data_{i}.RData"))
  Y <- simdata %>% dplyr::select(contains("Y."))
  Y_out <- Y %>% tail(400)
  Y_out_arr <- 1:2500 %>% lapply(\(i) Y_out) %>% abind::abind(along=3)
  coords_all <- simdata %>% dplyr::select(contains("coords")) 
  cx_S <- coords_all %>% head(3000) %>% sample_n(300, replace=F) %>% as.matrix()
  testx <- coords_all %>% tail(400) %>% as.matrix()
  
  theta_all <- rbind(20, 1, optlist, 1e-5)
  Sfact <- matrix(0, q, q)
  for(r in 1:q){
    cat("\n", r, "\n")
    for(j in 1:q){
      cat(j, " ")
      theta1 <- theta_all[,r]
      theta2 <- theta_all[,j]
      Sfact[r,j] <- scaling_factor_at_zero(testx, cx_S, theta1, theta2)
    }
  }
  Cov_at_zero <- Sigma * Sfact
  
  load(g("simulations/spiox_m/spiox_metrop_{i}.RData"))
  time_metrop <- total_time
  load(g("simulations/spiox_m/spiox_gibbs_{i}.RData"))
  time_gibbs <- total_time
  load(g("simulations/spiox_m/spiox_clust_{i}.RData"))
  time_clust <- total_time
  load(g("simulations/spiox_m/meshed_{i}.RData"))
  
  
  Y_out_gdiff <- rowSums(Y_out[,1:12]) - rowSums(Y_out[,13:24])
  target <- spiox_clust_predicts
  
  calc_performance <- function(target, Y){
    perf1 <- 1:q %>% sapply(\(i) cor( apply(target$Y[,i,], 1, mean), Y_out[,i] ))
    
    target_gdiff <- apply(target$Y[,1:12,], c(1,3), sum) - apply(target$Y[,13:24,], c(1,3), sum)
    perf2 <- cor( apply(target_gdiff, 1, mean), Y_out_gdiff )
    
    return(list(marginal_correlations = perf1,
                gdiff_correlation = perf2))
  }
  
  table_marginals <- \(listx){
    listx %>% lapply(\(x) x$marginal_correlations) %>% do.call(cbind, .)
  }
  
  table_gdiff <- \(listx){
    listx %>% lapply(\(x) x$gdiff_correlation) %>% do.call(cbind, .)
  }
  
  perf_spiox_gibbs <- calc_performance(spiox_gibbs_predicts)
  perf_spiox_metrop <- calc_performance(spiox_metrop_predicts)
  perf_spiox_clust <- calc_performance(spiox_clust_predicts)
  
  
  meshed_outdata <- simdata %>% mutate(sample = c(rep("insample", 3000), rep("outsample", 400))) %>% 
    left_join(spmeshed_out$coordsdata %>% mutate(ix=1:3400), by=c("coords.Var1"="Var1", "coords.Var2"="Var2"))
  
  meshed_outsample <- meshed_outdata %>%
    dplyr::filter(sample=="outsample") %$% ix
  
  Y_target_meshed <- meshed_outdata %>% dplyr::select(contains("Y.")) %>% `[`(meshed_outsample,) 
  meshed_predicts <- list(Y = spmeshed_out$yhat_mcmc %>% abind::abind(along=3) %>% `[`(meshed_outsample,,))
  
  perf_spmeshed <- calc_performance(meshed_predicts, Y_target_meshed)
  
  omega_spmeshed <- 1:2500 %>% lapply(\(m) with(spmeshed_out, 
              cov2cor(tcrossprod(lambda_mcmc[,,m]) + diag(tausq_mcmc[,m])))) %>%
    abind::abind(along=3) %>% apply(1:2, mean)
  
  
  results_gdiff <- table_gdiff(list(perf_spiox_gibbs, perf_spiox_metrop, perf_spiox_clust, perf_spmeshed))
  results_margs <- table_marginals(list(perf_spiox_gibbs, perf_spiox_metrop, perf_spiox_clust, perf_spmeshed))
  
  results[[i]] <- list(s=i, gdiff=results_gdiff, margs=results_margs)
}

save(file="simulations/multi_spiox_results.RData", list=c("results"))
