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

par_opts <- expand.grid(phi1 <- c(5, 15), phi2 <- phi1, rhocorr <- c(-0.9, -.5, 0.25, 0.75))
nsim <- 5

s <- 2
oo <- 1


for(s in 1:nrow(par_opts)){
  for(oo in 1:nsim){
    
    sim_n <- (s-1) * nsim + oo
    set.seed(sim_n)
    
    cat(sim_n, "\n") 
    
    
    philist <- par_opts[s, c("Var1", "Var2")] %>% as.vector()
    rhocorr <- par_opts[s, "Var3"]
      
    
    # spatial
    cx_in <- matrix(runif(2500*2), ncol=2)
    colnames(cx_in) <- c("Var1","Var2")
    n_in <- nrow(cx_in)
    which_in <- 1:n_in
    
    xout <- seq(0, 1, length.out=20)
    coords_out <- expand.grid(xout, xout)
    cx_out <- as.matrix(coords_out)
    n_out <- nrow(cx_out)
    which_out <- (n_in+1):(n_in+n_out)
    
    cx_all <- rbind(cx_in, cx_out)
    nr_all <- nrow(cx_all)
    
    q <- 2
    
    Clist <- philist %>% lapply(\(phi) spiox::Correlationc(cx_all, cx_all, c(phi,1,1,0), 1, TRUE) )
    Llist <- Clist %>% lapply(\(C) t(chol(C)))
    #Lilist <- Llist %>% lapply(\(L) solve(L))
    
    Sigma <- cbind(c(1, rhocorr), c(rhocorr, 1)) 
    St <- chol(Sigma)
    
    W <- Llist %>% lapply(\(L) L %*% rnorm(nr_all)) %>% do.call(cbind, .)
    Sigma <- cbind(c(1, rhocorr), c(rhocorr, 1)) 
    St <- chol(Sigma)
  
    # lmc
    Y_sp <- W %*% St
    
    # regression
    p <- 2
    X <- matrix(1, ncol=1, nrow=nr_all) %>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))
    
    Beta <- matrix(rnorm(q * p), ncol=q)
    
    Y_regression <- X %*% Beta
    #Error <- matrix(rnorm(nr * q),ncol=q) %*% diag(D <- runif(q, 0, 0.1))
    Y <- as.matrix(Y_sp + Y_regression) #+ Error
    
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
    
    
    save(file=glue::glue("simulations/lmc/data_{sim_n}.RData"), 
         list=c("simdata", "s", "oo", "sim_n", "simdata", "Sigma", "philist", "rhocorr"))
    
    ##############################
    
    set.seed(1)
    
    theta_opts <- cbind(c(10, 1, 1, 1e-19), c(20, 1, 1, 1e-19))
    ##############################################
  
    m_nn <- 20
    mcmc <- 5000
    
    if(F){
      custom_dag <- dag_vecchia(cx_in, m_nn)
      
      ##############################################
      set.seed(1) 
      estim_time <- system.time({
        spiox_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                          custom_dag = custom_dag, 
                                          theta=theta_opts,
                                          
                                          Sigma_start = diag(q),#Sigma,
                                          mvreg_B_start = 0*Beta,# %>% perturb(),
                                          
                                          mcmc = mcmc,
                                          print_every = 100,
                                          
                                          sample_iwish=T,
                                          sample_mvr=T,
                                          sample_theta_gibbs=F,
                                          upd_theta_opts=T,
                                          num_threads = 16)
      })
      
      predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
      
      predict_time <- system.time({
        spiox_predicts <- spiox::spiox_predict(X_new = X[which_out,],
                                               coords_new = cx_all[which_out,],
                                               
                                               # data
                                               Y_in, X_in, cx_in, 
                                               predict_dag,
                                               spiox_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$Sigma %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                               num_threads = 16)
      })
      
      Ytest <- spiox_predicts$Y %>% apply(1:2, mean)
      Ytrue <- Y[which_out,]
      1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))
      
      sqrt(mean( (Ytest[,1]*Ytest[,2] - Ytrue[,1]*Ytrue[,2])^2 ))
      
      total_time <- estim_time + predict_time
      
      save(file=glue::glue("simulations/lmc/spiox_{sim_n}.RData"), 
           list=c("spiox_out", "spiox_predicts", "estim_time", "predict_time", "total_time"))
    }
    
    if(F){
      # meshed
      library(meshed)
      
      Y_meshed <- Y
      Y_meshed[which_out,] <- NA
      
      meshed_time <- system.time({
        spmeshed_out <- meshed::spmeshed(y=Y_meshed, x=X, coords=cx_all, k=2, family = "gaussian",
                                       block_size=40, n_samples = mcmc, n_thin = 1, n_burn = 0, n_threads = 16, verbose = 10,
                                       predict_everywhere = T, prior = list(phi=c(.1, 20), tausq=c(1e-4,1e-4), nu=c(0.9999, 1.0001)))
      })
      
      m_order <- order(spmeshed_out$savedata$coords_blocking$ix)
      Ymesh_out <- spmeshed_out$yhat_mcmc %>% tailh() %>% abind::abind(along=3) %>% `[`(m_order[which_out],,)
      
      #Ymesh_mean <- Ymesh_out %>% apply(1:2, mean)
      #1:q %>% sapply(\(j) cor(Ymesh_mean[,j], Ytrue[,j]))
      #sqrt(mean( (Ymesh_mean[,1]*Ymesh_mean[,2] - Ytrue[,1]*Ytrue[,2])^2 ))
      
      save(file=glue::glue("simulations/lmc/meshed_{sim_n}.RData"), 
           list=c("spmeshed_out", "Ymesh_out", "meshed_time"))
      
    }
  
  }
}


