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



par_opts <- expand.grid(nu1 <- c(.75, 1.5), nu2 <- nu1, rhocorr <- c(-0.9, -.5, 0.25, 0.75))
nsim <- 5

s <- 1
oo <- 1


for(s in 1:nrow(par_opts)){
  for(oo in 1:nsim){
    
    sim_n <- (s-1) * nsim + oo
    set.seed(sim_n)
    
    cat(sim_n, "\n") 
    
    
    nulist <- par_opts[s, c("Var1", "Var2")] %>% as.vector()
    nu1 <- nulist$Var1
    nu2 <- nulist$Var2
    rhocorr <- par_opts[s, "Var3"]
      
    
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
    
    q <- 2
    
    cx_12 <- rbind(cbind(cx_all, 1), cbind(cx_all, 2))
    
    V <- cbind(c(1, rhocorr), c(rhocorr, 1)) 
    St <- chol(V)
    
    phi <- 20
    sigma11 <- 1
    sigma22 <- 1
    sigma12 <- sqrt(sigma11 * sigma22) * V[2,1] * 
      sqrt(gamma(nu1+1))/sqrt(gamma(nu1)) * 
      sqrt(gamma(nu2+1))/sqrt(gamma(nu2)) * gamma(nu1/2 + nu2/2)/gamma(nu1/2 + nu2/2 + 1)
    
    # parsimonious multi matern
    Cparmat <- GpGpm::matern_multi(c(sigma11, sigma12, sigma22, 
                                     1/phi, 1/phi, 1/phi, nu1, nu1/2+nu2/2, nu2, 0, 0, 0), cx_12)
    
    Lparmat <- t(chol(Cparmat))
    
    wvec <- Lparmat %*% rnorm(nr_all*2)
    W <- matrix(wvec, ncol=2)
    
    # matern
    Y_sp <- W 
    
    # regression
    p <- 1
    X <- matrix(1, ncol=1, nrow=nr_all) #%>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))
    
    Beta <- matrix(rnorm(q * p), ncol=q)
    
    Y_regression <- X %*% Beta
    Error <- matrix(rnorm(nrow(Y_regression) * q),ncol=q) %*% diag(D <- runif(q, 0, 0.1))
    Y <- as.matrix(Y_sp + Y_regression) + Error
    
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
    
    save(file=glue::glue("simulations/matern2/data_{sim_n}.RData"), 
         list=c("simdata", "s", "oo", "sim_n",
                "Beta", "D",
                "simdata", "sigma12", "nulist", "rhocorr"))
    
    ##############################
    
    set.seed(1)
    
    #theta_opts <- cbind(c(10, 1, .5, 1e-19), c(20, 1, 1, 1e-19))
    theta_opts <- cbind(c(20, 1, nu1, 1e-1), c(20, 1, nu2, 1e-2))
    ##############################################
  
    m_nn <- 20
    mcmc <- 5000
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
    
    if(T){
      custom_dag <- dag_vecchia(cx_in, m_nn)
      
      ##############################################
      set.seed(1) 
      estim_time <- system.time({
        spiox_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                          custom_dag = custom_dag, 
                                          theta=theta_opts,
                                          
                                          Sigma_start = diag(q),
                                          mvreg_B_start = 0*Beta,# %>% perturb(),
                                          
                                          mcmc = mcmc,
                                          print_every = 100,
                                          matern = TRUE,
                                          sample_iwish=T,
                                          sample_mvr=T,
                                          sample_theta_gibbs=F,
                                          upd_theta_opts=T,
                                          num_threads = 16)
      })
      
      predict_dag <- dag_vecchia_predict(cx_in, cx_all[which_out,], m_nn)
      
      predict_time <- system.time({
        spiox_predicts <- spiox::spiox_predict(X_new = X[which_out,,drop=F],
                                               coords_new = cx_all[which_out,],
                                               
                                               # data
                                               Y_in, X_in, cx_in, 
                                               predict_dag,
                                               spiox_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$Sigma %>% tail(c(NA, NA, round(mcmc/2))), 
                                               spiox_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                                               matern = TRUE,
                                               num_threads = 16)
      })
      

      total_time <- estim_time + predict_time
      
      save(file=glue::glue("simulations/matern2/spiox_{sim_n}.RData"), 
           list=c("spiox_out", "spiox_predicts", "estim_time", "predict_time", "total_time"))
    }
    
    if(T){
      # meshed
      library(meshed)
      
      Y_meshed <- Y
      Y_meshed[which_out,] <- NA
      
      meshed_time <- system.time({
        spmeshed_out <- meshed::spmeshed(y=Y_meshed, x=X, coords=cx_all, k=2, family = "gaussian",
                                       block_size=40, n_samples = mcmc, n_thin = 1, n_burn = 0, n_threads = 16, verbose = 10,
                                       predict_everywhere = T, prior = list(phi=c(.1, 20), tausq=c(1e-4,1e-4), nu=c(.5, 1.9)))
      })
      
      m_order <- order(spmeshed_out$savedata$coords_blocking$ix)
      Ymesh_out <- spmeshed_out$yhat_mcmc %>% tailh() %>% abind::abind(along=3) %>% `[`(m_order[which_out],,)
      
      save(file=glue::glue("simulations/matern2/meshed_{sim_n}.RData"), 
           list=c("spmeshed_out", "Ymesh_out", "meshed_time"))
      
    }
  
  }
}


