args <- commandArgs(TRUE)

starts <- args[1]
ends <- args[2]

cat("Simulations ", starts:ends, "\n")

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

nthreads <- 8

for(oo in starts:ends){
  set.seed(oo)
  cat(oo, "\n") 
  
  optlist <- seq(5, 50, length.out=q) #%>% sample(q, replace=T)
  
  # spatial
  cx_in <- matrix(runif(2500*2), ncol=2) #3000
  colnames(cx_in) <- c("Var1","Var2")
  n_in <- nrow(cx_in)
  which_in <- 1:n_in
  
  xout <- seq(0, 1, length.out=20) #20
  coords_out <- expand.grid(xout, xout)
  cx_out <- as.matrix(coords_out)
  n_out <- nrow(cx_out)
  which_out <- (n_in+1):(n_in+n_out)
  
  cx_all <- rbind(cx_in, cx_out)
  nr_all <- nrow(cx_all)
  
  Clist <- optlist %>% lapply(\(phi) spiox::Correlationc(cx_all, cx_all, c(phi,1,1,1e-15), 1, TRUE) )
  Llist <- Clist %>% lapply(\(C) t(chol(C)))
  
  Q <- rWishart(1, q+1, 1/2 * diag(q))[,,1] #
  Sigma <- solve(Q) 
  St <- chol(Sigma)
  S <- t(St)
  
  V <- matrix(rnorm(nr_all * q), ncol=q) 
  V <- V
  for(i in 1:q){
    V[,i] <- Llist[[i]] %*% V[,i]
  }
  Y_sp <- V %*% St
  
  # regression
  p <- 2
  X <- matrix(1, ncol=1, nrow=nr_all) %>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))
  
  Beta <- matrix(rnorm(q * p), ncol=q)
  
  Y_regression <- X %*% Beta
  
  Y <- as.matrix(Y_sp + Y_regression) + matrix(rnorm(nr_all * q), ncol=q) %*% diag(rep(0.01, q))
  
  Y_in <- Y[which_in,]
  X_in <- X[which_in,]
  
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
  
  save(file=glue::glue("simulations/lmc_m/data_{oo}.RData"), 
       list=c("simdata", "oo", "simdata", "Sigma", "optlist"))
  
  ##############################
  
  set.seed(1)
  
  nutsq <- expand.grid(nu <- seq(0.5, 2, length.out=20),
                       tsq <- c(2*1e-6, 1e-5, 1e-4))
  
  theta_opts <- rbind(20, 1, nutsq$Var1, nutsq$Var2)
  ##############################################
  
  m_nn <- 20
  mcmc <- 5000
  
  if(F){
    custom_dag <- dag_vecchia(cx_in, m_nn)
    
    ##############################################
    set.seed(1) 
    RhpcBLASctl::omp_set_num_threads(1)
    RhpcBLASctl::blas_set_num_threads(1)
    estim_time <- system.time({
      spiox_gibbs_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                              custom_dag = custom_dag, 
                                              theta=theta_opts,
                                              
                                              Sigma_start = diag(q),
                                              mvreg_B_start = 0*Beta,# %>% perturb(),
                                              
                                              mcmc = mcmc,
                                              print_every = 100,
                                              
                                              sample_iwish=T,
                                              sample_mvr=T,
                                              sample_theta_gibbs=T,
                                              upd_theta_opts=F,
                                              num_threads = nthreads)
    })
    
    
    
    predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
    
    predict_time <- system.time({
      spiox_gibbs_predicts <- spiox::spiox_predict(X_new = X[which_out,],
                                                   coords_new = cx_all[which_out,],
                                                   
                                                   # data
                                                   Y_in, X_in, cx_in, 
                                                   predict_dag,
                                                   spiox_gibbs_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   spiox_gibbs_out$S %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   spiox_gibbs_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   num_threads = nthreads)
    })
    
    Ytest <- spiox_gibbs_predicts$Y %>% stailh() %>% apply(1:2, mean)
    Ytrue <- Y[which_out,]
    1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))
    
    Y_spiox_sum_post_mean <- with(spiox_gibbs_predicts, apply(Y[,1,]+Y[,2,], 1, mean))
    sqrt(mean( (Y_spiox_sum_post_mean - Ytrue[,1]-Ytrue[,2])^2 ))
    
    total_time <- estim_time + predict_time
    
    save(file=glue::glue("simulations/lmc_m/spiox_gibbs_{oo}.RData"), 
         list=c("spiox_gibbs_out", "spiox_gibbs_predicts", "estim_time", "predict_time", "total_time"))
    rm(list=c("spiox_gibbs_out", "spiox_gibbs_predicts"))
  }
  
  if(T){
    custom_dag <- dag_vecchia(cx_in, m_nn)
    
    ##############################################
    set.seed(1) 
    RhpcBLASctl::omp_set_num_threads(1)
    RhpcBLASctl::blas_set_num_threads(1)
    estim_time <- system.time({
      spiox_metrop_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                               custom_dag = custom_dag, 
                                               theta=theta_opts[,1:q],
                                               
                                               Sigma_start = diag(q),
                                               mvreg_B_start = 0*Beta,# %>% perturb(),
                                               
                                               mcmc = mcmc,
                                               print_every = 100,
                                               
                                               sample_iwish=T,
                                               sample_mvr=T,
                                               sample_theta_gibbs=F,
                                               upd_theta_opts=T,
                                               num_threads = nthreads)
    })
    
    
    
    predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
    
    predict_time <- system.time({
      spiox_metrop_predicts <- spiox::spiox_predict(X_new = X[which_out,],
                                                    coords_new = cx_all[which_out,],
                                                    
                                                    # data
                                                    Y_in, X_in, cx_in, 
                                                    predict_dag,
                                                    spiox_metrop_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                                    spiox_metrop_out$S %>% tail(c(NA, NA, round(mcmc/2))), 
                                                    spiox_metrop_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                                    num_threads = nthreads)
    })
    
    Ytest <- spiox_metrop_predicts$Y %>% stailh() %>% apply(1:2, mean)
    Ytrue <- Y[which_out,]
    1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))
    
    Y_spiox_sum_post_mean <- with(spiox_metrop_predicts, apply(Y[,1,]+Y[,2,], 1, mean))
    sqrt(mean( (Y_spiox_sum_post_mean - Ytrue[,1]-Ytrue[,2])^2 ))
    
    total_time <- estim_time + predict_time
    
    save(file=glue::glue("simulations/lmc_m/spiox_metrop_{oo}.RData"), 
         list=c("spiox_metrop_out", "spiox_metrop_predicts", "estim_time", "predict_time", "total_time"))
    
    rm(list=c("spiox_metrop_out", "spiox_metrop_predicts"))
  }
  
  if(T){
    custom_dag <- dag_vecchia(cx_in, m_nn)
    
    ##############################################
    set.seed(1) 
    RhpcBLASctl::omp_set_num_threads(1)
    RhpcBLASctl::blas_set_num_threads(1)
    estim_time <- system.time({
      spiox_clust_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                              custom_dag = custom_dag, 
                                              theta=theta_opts[,1:6],
                                              
                                              Sigma_start = diag(q),
                                              mvreg_B_start = 0*Beta,# %>% perturb(),
                                              
                                              mcmc = mcmc,
                                              print_every = 100,
                                              
                                              sample_iwish=T,
                                              sample_mvr=T,
                                              sample_theta_gibbs=T,
                                              upd_theta_opts=T,
                                              num_threads = nthreads)
    })
    
    
    
    predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
    
    predict_time <- system.time({
      spiox_clust_predicts <- spiox::spiox_predict(X_new = X[which_out,],
                                                   coords_new = cx_all[which_out,],
                                                   
                                                   # data
                                                   Y_in, X_in, cx_in, 
                                                   predict_dag,
                                                   spiox_clust_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   spiox_clust_out$S %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   spiox_clust_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                                   num_threads = nthreads)
    })
    
    Ytest <- spiox_clust_predicts$Y %>% stailh() %>% apply(1:2, mean)
    Ytrue <- Y[which_out,]
    1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))
    
    Y_spiox_sum_post_mean <- with(spiox_clust_predicts, apply(Y[,1,]+Y[,2,], 1, mean))
    sqrt(mean( (Y_spiox_sum_post_mean - Ytrue[,1]-Ytrue[,2])^2 ))
    
    total_time <- estim_time + predict_time
    
    save(file=glue::glue("simulations/lmc_m/spiox_clust_{oo}.RData"), 
         list=c("spiox_clust_out", "spiox_clust_predicts", "estim_time", "predict_time", "total_time"))
    rm(list=c("spiox_clust_out", "spiox_clust_predicts"))
  }
  
  if(F){
    # meshed
    library(meshed)
    
    Y_meshed <- Y
    Y_meshed[which_out,] <- NA
    
    meshed_time <- system.time({
      spmeshed_out <- meshed::spmeshed(y=Y_meshed, x=X, coords=cx_all, k=6, family = "gaussian",
                                       block_size=40, 
                                       n_samples = round(mcmc/2), n_thin = 1, n_burn = round(mcmc/2), 
                                       n_threads = nthreads, verbose = 10,
                                       predict_everywhere = T, 
                                       prior = list(phi=c(.1, 50), tausq=c(1e-4,1e-4), nu=c(.5, 2)))
    })
    
    m_order <- order(spmeshed_out$savedata$coords_blocking$ix)
    Ymesh_out <- spmeshed_out$yhat_mcmc %>% tailh() %>% abind::abind(along=3) %>% `[`(m_order[which_out],,)
    
    Ymesh_mean <- Ymesh_out %>% apply(1:2, mean)
    1:q %>% sapply(\(j) cor(Ymesh_mean[,j], Ytrue[,j]))
    Y_meshed_sum_post_mean <- apply(Ymesh_out[,1,]+Ymesh_out[,2,], 1, mean)
    sqrt(mean( (Y_meshed_sum_post_mean - Ytrue[,1]-Ytrue[,2])^2 ))
    
    save(file=glue::glue("simulations/lmc_m/meshed_{oo}.RData"), 
         list=c("spmeshed_out", "meshed_time"))
    
  }
  
}


