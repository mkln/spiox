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

par_opts <- expand.grid(nu1 <- c(0.75, 1.5), nu2 <- nu1, rhocorr <- c(-0.9, -.5, 0.25, 0.75))
nsim <- 5

s <- 2
oo <- 1


for(s in 1:nrow(par_opts)){
  for(oo in 1:nsim){
    
    sim_n <- (s-1) * nsim + oo
    set.seed(sim_n)
    
    cat(sim_n, "\n") 
    nulist <- par_opts[s, c("Var1", "Var2")] %>% as.vector()
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
    
    D <- runif(q, 0, 0.1)
    
    Clist <- 1:q %>% lapply(\(j) spiox::Correlationc(cx_all, cx_all, c(20,1,nulist[[j]],D[j]), 1, TRUE) )
    Llist <- Clist %>% lapply(\(C) t(chol(C)))
    Lilist <- Llist %>% lapply(\(L) solve(L))
    
    Sigma <- cbind(c(1, rhocorr), c(rhocorr, 1)) 
    St <- chol(Sigma)
    
    V <- matrix(rnorm(nr_all * q), ncol=q) %*% St
    Sigma <- cbind(c(1, rhocorr), c(rhocorr, 1)) 
    St <- chol(Sigma)
    
    
    Y_sp <- V
    # make q spatial outcomes
    for(i in 1:q){
      Y_sp[,i] <- Llist[[i]] %*% V[,i]
    }
    
    # regression
    p <- 1
    X <- matrix(1, ncol=1, nrow=nr_all)# %>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))
    
    Beta <- matrix(rnorm(q * p), ncol=q)
    
    Y_regression <- X %*% Beta
    
    Y <- as.matrix(Y_sp + Y_regression) #+ Error
    
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
    
    save(file=glue::glue("simulations/spiox/data_{sim_n}.RData"), 
         list=c("simdata", "s", "oo", "sim_n", "simdata",
                "D", "Sigma", "nulist", "rhocorr"))
    
    ##############################
    
    set.seed(1)
    
    theta_opts <- cbind(c(20, 1, .5, 1e-2), c(20, 1, 1.5, 1e-3))
    theta_opts_latent <- cbind(c(20, 1, 0.51, 1e-10), c(20, 1, 1.5, 1e-10))
    ##############################################
  
    m_nn <- 20
    mcmc <- 5000
    
    if(T){
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
      
      set.seed(1)
      predict_time <- system.time({
        spiox_predicts <- spiox::spiox_predict(X_new = X[which_out,,drop=F],
                                               coords_new = cx_all[which_out,],
                                               
                                               # data
                                               Y_in, X_in, cx_in, 
                                               predict_dag,
                                               spiox_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$Sigma %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                               num_threads = 16)
      })
      
      total_time <- estim_time + predict_time
      
      save(file=glue::glue("simulations/spiox/spiox_{sim_n}.RData"), 
           list=c("spiox_out", "spiox_predicts", "estim_time", "predict_time", "total_time"))
    }
    
    
    if(T){
      custom_dag <- dag_vecchia(cx_in, m_nn)
      
      ##############################################
      set.seed(1) 
      latent_estim_time <- system.time({
        spiox_latent_out <- spiox::spiox_latent(Y_in, X_in, cx_in, 
                                          custom_dag = custom_dag, 
                                          theta=theta_opts_latent,
                                          
                                          Sigma_start = diag(q),#Sigma,
                                          mvreg_B_start = matrix(0, p, q),#Beta,# %>% perturb(),
                                          
                                          mcmc = mcmc,
                                          print_every = 100,
                                          
                                          sample_iwish=T,
                                          sample_mvr=T,
                                          sample_theta_gibbs=F,
                                          upd_theta_opts=T,
                                          num_threads = 16, 
                                          sampling = 2)
      })
      
      predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
      
      latent_predict_time <- system.time({
        spiox_latent_predicts <- 
          spiox_latent_out %>% with(
                   spiox::spiox_latent_predict(X_new = X[which_out,,drop=F],
                                               coords_new = cx_all[which_out,],
                                               
                                               # data
                                               cx_in, 
                                               predict_dag,
                                               W %>% tail(c(NA, NA, round(mcmc/2))),
                                               B %>% tail(c(NA, NA, round(mcmc/2))), 
                                               Sigma %>% tail(c(NA, NA, round(mcmc/2))), 
                                               Ddiag %>% tail(c(NA, round(mcmc/2))),
                                               theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                               num_threads = 16) )
      })
      
      Ytest <- spiox_latent_predicts$Y %>% apply(1:2, mean)
      Ytrue <- Y[which_out,]
      1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))
      
      spiox_latent_predicts %>% with(Y[,1,]+Y[,2,]) %>% apply(1, mean) %>% cbind(Ytrue[,1]+Ytrue[,2]) %>% cor
      
      latent_total_time <- estim_time + predict_time
      
      save(file=glue::glue("simulations/spiox/spioxlat_{sim_n}.RData"), 
           list=c("spiox_latent_out", "spiox_latent_predicts", 
                  "latent_estim_time", "latent_predict_time", "latent_total_time"))
    }
    
    if(T){
      # nngp
      library(spNNGP)
      
      starting <- list("phi"=20, "sigma.sq"=1, "tau.sq"=1e-19, "nu"=1)
      tuning <- list("phi"=0, "sigma.sq"=0, "tau.sq"=0.1, "nu"=0.1)
      priors.1 <- list("beta.Norm"=list(rep(0,ncol(X_in)), diag(1e3,ncol(X_in))),
                       "phi.Unif"=c(1, 60), "sigma.sq.IG"=c(2, 1),
                       "nu.Unif"=c(0.5, 1.99),
                       "tau.sq.IG"=c(1e-3, 1e-3))
      
      verbose <- TRUE
      n.neighbors <- m_nn
      burnin <- 1:round(mcmc/2)
      
      nngp_time <- system.time({
        m.s.1 <- spNNGP::spNNGP(Y_in[,1] ~ X_in - 1, 
                                coords=cx_in, 
                                starting=starting, method="response", 
                                n.neighbors=n.neighbors,
                                tuning=tuning, priors=priors.1, cov.model="matern", 
                                n.samples=mcmc, n.omp.threads=16)
        
        m.s.1$p.beta.samples %<>% `[`(-burnin,,drop=F)
        m.s.1$p.theta.samples %<>% `[`(-burnin,,drop=F)
        
        nngp_pred.1 <- predict(m.s.1, X[which_out,,drop=F], 
                               coords=cx_out, n.omp.threads=16)
        
        m.s.2 <- spNNGP::spNNGP(Y_in[,2] ~ X_in - 1, 
                                coords=cx_in, 
                                starting=starting, method="response", 
                                n.neighbors=n.neighbors,
                                tuning=tuning, priors=priors.1, cov.model="matern", 
                                n.samples=mcmc, n.omp.threads=16)
        
        m.s.2$p.beta.samples %<>% `[`(-burnin,,drop=F)
        m.s.2$p.theta.samples %<>% `[`(-burnin,,drop=F)
        
        nngp_pred.2 <- predict(m.s.2, X[which_out,,drop=F], 
                               coords=cx_out, n.omp.threads=16)
      })
      
      save(file=glue::glue("simulations/spiox/nngp_{sim_n}.RData"), 
           list=c("burnin", "m.s.1", "nngp_pred.1", "m.s.2", "nngp_pred.2", "nngp_time"))
      
      
    }
  
  }
}


