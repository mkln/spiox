rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0, 1, length.out=30)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

q <- 3

k <- 3
optlist <- seq(3, 20, length.out=q) %>% sample(k, replace=T)

Clist <- optlist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1) + 1e-16*diag(nr)) )
#Clist <- optlist %>% lapply(\(nu) spiox::Correlationc(cx, cx, c(20,1,nu,1e-8), 1, TRUE) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))

Q <- rWishart(1, q+2, diag(q))[,,1] #
Sigma <- solve(Q) 

St <- chol(Sigma)
S <- t(St)

W <- 1:k %>% lapply(\(j) Llist[[j]] %*% rnorm(nr)) %>% abind::abind(along=2)

Y_sp <- W %*% St

# regression
p <- 2
X <- matrix(1, ncol=1, nrow=nr) %>% cbind(matrix(rnorm(nr*(p-1)), ncol=p-1))

Beta <- matrix(rnorm(q * p), ncol=q)

Y_regression <- X %*% Beta
Error <- matrix(rnorm(nr * q),ncol=q) # %*% diag(D <- runif(q, 0, 0.1))
Y <- as.matrix(Y_sp + Y_regression) #+ Error

cov_Y_cols <- cov(Y) %>% as("dgCMatrix")
cov_Y_rows <- cov(t(Y)) %>% as("dgCMatrix")

df <- data.frame(coords, y=Y_sp) %>% 
  pivot_longer(cols=-c(Var1, Var2))

if(F){
  ggplot(df, 
         aes(Var1, Var2, fill=value)) +
    geom_raster() +
    scale_fill_viridis_c() +
    facet_wrap(~name, ncol=5)
}

##############################

set.seed(1)

radgp_rho <- .15
test_radgp <- spiox::radgp_build(cx, radgp_rho, phi=10, sigmasq=1, nu=1.5, tausq=0, matern=F)
test_radgp$dag %>% sapply(\(x) nrow(x)) %>% summary()

par_opts <- seq(5, 30, length.out=k)
theta_opts <- par_opts %>% sapply(\(phi) matrix( c(phi, 1, 1, 1e-16), ncol=1))

testset <- sample(1:nrow(Y), 10, replace=F)

perturb <- function(x, sd=1){
  return(x + matrix(rnorm(prod(dim(x)), sd), ncol=ncol(x)))
}

set.seed(1) 
(total_time <- system.time({
  spiox_out <- spiox::spiox_wishart(Y[-testset,,drop=F], 
                                    X[-testset,,drop=F],
                                    cx[-testset,], 
                                    radgp_rho = radgp_rho, theta=theta_opts,
                                    
                                    Sigma_start = diag(q),
                                    mvreg_B_start = 0*Beta,# %>% perturb(),
                                    
                                    mcmc = mcmc <- 5000,
                                    print_every = 50,
                                    
                                    sample_iwish=T,
                                    sample_mvr=T,
                                    sample_gp=T,
                                    upd_opts=T,
                                    num_threads = 16)
}))

Sig_est <- 1:mcmc %>% sapply(\(m) with(spiox_out,  diag(sqrt(theta[2,,m])) %*% 
                               Sigma[,,m] %*% diag(sqrt(theta[2,,m])) ) ) %>% 
  array(c(q,q,mcmc)) %>% apply(1:2, mean)

thetamean <- spiox_out$theta %>% apply(1:2, mean)
philist <- thetamean[1,]
siglist <- thetamean[2,]

hvec <- c(seq(0, 0.008, length.out=10), seq(0.01, .5, length.out=10))
xg <- seq(0.1, 0.9, length.out=5)
test_coords <- expand.grid(xg,xg) %>% as.matrix()


i <- 2; j <- 2
testh <- Sig_est[i,j] * spiox::iox_cross_avg(hvec, i, j,
                                             test_coords, cx, 
                                             philist, c(1,1,1), 12, 1, num_threads=15) 

truth <- hvec %>% sapply(\(h) (S %*% diag(exp(-optlist*h)) %*% t(S))[i,j])

mesh_total_time <- system.time({
  meshout <- meshed::spmeshed(y=Y, 
                      x=X,
                      coords=cx, k = q,
                      block_size = 50, 
                      n_samples = 10000, 
                      n_burn = 1, 
                      n_thin = 1, 
                      n_threads = 16,
                      prior = list(phi=c(1, 20)),
                      verbose=50
  )})
h <- 0
mesh_mcmc <- dim(meshout$theta_mcmc)[3]

mesh_h_f <- function(h){
  sig_mcmc <- 1:mesh_mcmc %>% sapply(\(m) with(meshout, 
                                                       lambda_mcmc[,,m] %*% diag(exp(-theta_mcmc[1,,m]*h)) %*% t(lambda_mcmc[,,m]) + diag(tausq_mcmc[,m])
  )) %>% array(dim=c(3,3,mesh_mcmc))
  return(sig_mcmc %>% apply(1:2, mean))
}



phi_mesh_out <- meshout$theta_mcmc[1,,] %>% apply(1, mean)

Sig_mesh_out <- meshout$lambda_mcmc %>% apply(3,\(l) tcrossprod(l)) %>% array(dim=c(q,q,mcmc))
Sig_mesh_est <- Sig_mesh_out %>% apply(1:2, mean)
S_mesh <- t(chol(Sig_mesh_est))




meshh <- hvec %>% sapply(\(h) mesh_h_f(h)[i,j])

plot(hvec, truth, type='l', ylim=c(-2,1))
lines(hvec, meshh, col="red")
lines(hvec, testh, col="blue")



set.seed(1)
(total_time2 <- system.time({
  spassso_out <- spassso::spassso(Y[-testset,], 
                                  X[-testset,,drop=F],
                                  cx[-testset,], 
                                  radgp_rho = radgp_rho, theta=theta_opts,
                                  
                                  spf_k = q, spf_a_delta = .1, spf_b_delta = .1, spf_a_dl = 0.9,
                                  
                                  spf_Lambda_start = matrix(rnorm(q*q), ncol=q) %>% perturb(),
                                  spf_Delta_start = runif(q),#diag(Delta),
                                  mvreg_B_start = Beta,# %>% perturb(),
                                  
                                  mcmc = mcmc <- 1000,
                                  print_every=100,
                                  
                                  sample_precision=1,
                                  sample_mvr=T,
                                  sample_gp=T)
}))

Sig_lmc <- spassso_out$S %>% apply(3, \(s) crossprod(s)) %>% array(dim=c(q,q,mcmc))
Sig_lmc_est <- Sig_lmc %>% apply(1:2, mean)

