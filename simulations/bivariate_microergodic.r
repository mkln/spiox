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


set.seed(1)

rhocorr <- 0.8

# spatial
cx_in <- as.matrix(expand.grid(xx <- seq(0, 1, length.out=50), xx))
colnames(cx_in) <- c("Var1","Var2")
n_in <- nrow(cx_in)
which_in <- 1:n_in

cx_all <- cx_in
nr_all <- nrow(cx_all)

q <- 2

matern <- FALSE
phis <- c(20, 30)
nus <- c(1, 1)
Clist <- 1:q %>% lapply(\(j) spiox::Correlationc(cx_all, cx_all, c(phis[j], 1, nus[j], 1e-15), matern, TRUE) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))

Sigma <- cbind(c(2, 1.2), c(1.2, 1.6))
St <- chol(Sigma)

set.seed(1)
V <- matrix(rnorm(nr_all * q), ncol=q) %*% St

Y_sp <- V
# make q spatial outcomes
for(i in 1:q){
  Y_sp[,i] <- Llist[[i]] %*% V[,i]
}

# regression
p <- 1
X <- matrix(1, ncol=1, nrow=nr_all)# %>% cbind(matrix(rnorm(nr_all*(p-1)), ncol=p-1))

Beta <- 0*matrix(rnorm(q * p), ncol=q)

Y_regression <- X %*% Beta

Y <- as.matrix(Y_sp + Y_regression) #+ Error

Y_in <- Y[which_in,]
X_in <- X[which_in,,drop=F]

if(F){
  df <- data.frame(cx_all, y=as.matrix(Y_sp)) %>% 
    pivot_longer(cols=-c(Var1, Var2))
  ggplot(df, 
         aes(Var1, Var2, fill=value)) +
    geom_raster() +
    scale_fill_viridis_c() +
    facet_wrap(~name, ncol=5)
}

set.seed(1)

theta_opts <- cbind(c(10, 1, 1, 1e-15), c(20, 2, 1, 1e-15))

##############################################

m_nn <- 10
mcmc <- 2500

custom_dag <- dag_vecchia(cx_in, m_nn)

##############################################
set.seed(1) 
estim_time <- system.time({
  spiox_exp_out <- spiox::spiox_wishart(Y_in, X_in, cx_in, 
                                    custom_dag = custom_dag, 
                                    theta=theta_opts,
                                    
                                    Sigma_start = cov(Y_in),#diag(q),#Sigma,
                                    mvreg_B_start = Y_in %>% apply(2, mean) %>% matrix(nrow=1), #0*Beta,# %>% perturb(),
                                    
                                    mcmc = mcmc,
                                    print_every = 100,
                                    matern = FALSE,
                                    sample_iwish=T,
                                    sample_mvr=T,
                                    sample_theta_gibbs=F,
                                    upd_theta_opts=T,
                                    num_threads = 2)
})

spiox_exp_out %>% with(theta[1,1,]) %>% plot(type='l')
spiox_exp_out %>% with(theta[1,2,]) %>% plot(type='l')

spiox_exp_out %>% with(Sigma[1,1,]) %>% plot(type='l')
spiox_exp_out %>% with(Sigma[2,2,]) %>% plot(type='l')

spiox_exp_out %>% with(Sigma[2,1,]) %>% plot(type='l')

spiox_exp_out %>% with(Sigma[1,1,] * theta[1,1,]) %>% plot(type='l')
spiox_exp_out %>% with(Sigma[2,2,] * theta[1,2,]) %>% plot(type='l')

Omega_mcmc <- spiox_exp_out %>% with(Sigma_to_correl(Sigma))
Omega_mcmc[2,1,] %>% plot(type='l')





########## squared exponential

matern <- FALSE
Clist <- 1:q %>% lapply(\(j) spiox::Correlationc(cx_all, cx_all, c(15, 1, 2, 1e-6), matern, TRUE) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))

Y_sp <- V
# make q spatial outcomes
for(i in 1:q){
  Y_sp[,i] <- Llist[[i]] %*% V[,i]
}

Y <- as.matrix(Y_sp + Y_regression) #+ Error

if(F){
  df <- data.frame(cx_all, y=as.matrix(Y_sp)) %>% 
    pivot_longer(cols=-c(Var1, Var2))
  ggplot(df, 
         aes(Var1, Var2, fill=value)) +
    geom_raster() +
    scale_fill_viridis_c() +
    facet_wrap(~name, ncol=5)
}

set.seed(1)

theta_opts <- cbind(c(20, 1, 2, 1e-6), c(10, 1, 2, 1e-6))

##############################################
set.seed(1) 
estim_time <- system.time({
  spiox_sqe_out <- spiox::spiox_wishart(Y, X, cx_in, 
                                    custom_dag = custom_dag, 
                                    theta=theta_opts,
                                    
                                    Sigma_start = cov(Y),#diag(q),#Sigma,
                                    mvreg_B_start = Y_in %>% apply(2, mean) %>% matrix(nrow=1), #0*Beta,# %>% perturb(),
                                    
                                    mcmc = mcmc,
                                    print_every = 100,
                                    matern = FALSE,
                                    sample_iwish=T,
                                    sample_mvr=T,
                                    sample_theta_gibbs=F,
                                    upd_theta_opts=T,
                                    num_threads = 2)
})

spiox_sqe_out %>% with(theta[1,1,]) %>% tail(9000) %>% plot(type='l')
spiox_sqe_out %>% with(theta[1,2,]) %>% tail(9000) %>% plot(type='l')
spiox_sqe_out %>% with(Sigma[1,1,]) %>% tail(2000) %>% plot(type='l')
spiox_sqe_out %>% with(Sigma[2,2,]) %>% tail(2000) %>% plot(type='l')






# meshed 

library(meshed)

meshed_time <- system.time({
  spmeshed_out <- meshed::spmeshed(y=Y, x=X, coords=cx_in, k=2, family = "gaussian",
                                   block_size=40, n_samples = mcmc, n_thin = 1, 
                                   n_burn = 0, n_threads = 2, verbose = 10,
                                   prior = list(phi=c(.1, 20), tausq=c(1e-4,1e-4)),
                                   settings=list(cache=TRUE, ps=FALSE))
})


Sigma_meshed <- spmeshed_out$lambda_mcmc %>% meshed:::cube_tcrossprod()
Omega_meshed <- Sigma_meshed %>% Sigma_to_correl()









