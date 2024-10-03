rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0, 1, length.out=50)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

test_radgp <- spiox::radgp_build(cx, 0.12, phi=20, sigmasq=1, nu=1.5, tausq=0, matern=T, 16)
w <- solve(test_radgp$H, rnorm(nr))

Ctest <- spiox::Correlationc(cx, cx, c(1,1,1.9,1-8), 1, TRUE)

q <- 2

optlist <- seq(0.4, 21.9, length.out=q) %>% sample(q, replace=T)

#Clist <- optlist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1) + 1e-6*diag(nr)) )
Clist <- optlist %>% lapply(\(nu) spiox::Correlationc(cx, cx, c(20,1,nu,1e-8), 1, TRUE) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Lilist <- Llist %>% lapply(\(L) solve(L))


Q <- rWishart(1, q+2, 1/20*diag(q))[,,1] #
Sigma <- solve(Q) 

St <- chol(Sigma)
S <- t(St)



V <- matrix(rnorm(nr * q), ncol=q) %*% St

Y_sp <- V

save_which <- rep(0, q)

# make q spatial outcomes
which_process <- rep(0, q)
for(i in 1:q){
  which_process[i] <- i #1:length(optlist) %>% sample(1)
  Y_sp[,i] <- Llist[[which_process[i]]] %*% V[,i]
}
theta_true <- optlist[which_process]

# regression
p <- 2
X <- matrix(1, ncol=1, nrow=nr) %>% cbind(matrix(rnorm(nr*(p-1)), ncol=p-1))

Beta <- matrix(rnorm(q * p), ncol=q)

Y_regression <- X %*% Beta
Error <- matrix(rnorm(nr * q),ncol=q) %*% diag(D <- runif(q, 0, 0.1))
Y <- as.matrix(Y_sp + Y_regression) + Error

cov_Y_cols <- cov(Y) %>% as("dgCMatrix")
cov_Y_rows <- cov(t(Y)) %>% as("dgCMatrix")

df <- data.frame(coords, y=as.matrix(Y_sp)) %>% 
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

radgp_rho <- .1
test_radgp <- spiox::radgp_build(cx, radgp_rho, phi=10, sigmasq=1, nu=1.5, tausq=0, matern=T)


test_radgp$dag %>% sapply(\(x) nrow(x)) %>% summary()

par_opts <- seq(0.51, 1.8, length.out=5)

theta_opts <- cbind(c(19, 1, 0.51, 1e-4), c(20, 1, 1.5, 1e-5))
#theta_opts <- par_opts %>% sapply(\(nu) matrix( c(20, 1, nu, 1e-6), ncol=1))
#theta_opts <- par_opts %>% sapply(\(phi) matrix( c(phi, 1, 1, 1e-16), ncol=1))

testset <- sample(1:nrow(Y), 10, replace=F)

perturb <- function(x, sd=1){
  return(x + matrix(rnorm(prod(dim(x)), sd), ncol=ncol(x)))
}

set.seed(1) 
(total_time <- system.time({
  spiox_out <- spiox::spiox_wishart(Y[-testset,,drop=F], 
                            X[-testset,,drop=F],
                            cx[-testset,], 
                            radgp_rho = radgp_rho, theta=theta_opts[,1:2],
                            
                            Sigma_start = diag(q),
                            mvreg_B_start = 0*Beta,# %>% perturb(),
                            
                            mcmc = mcmc <- 10000,
                            print_every = 100,
                            
                            sample_iwish=T,
                            sample_mvr=T,
                            sample_theta_gibbs=T,
                            upd_theta_opts=T,
                            num_threads = 16)
}))


boom2 <- spiox::spiox_predict(X_new = X[testset,,drop=F],
                              coords_new = cx[testset,],
                              
                              # data
                              Y[-testset,], 
                              X[-testset,,drop=F], 
                              cx[-testset,], 
                              
                              radgp_rho, 
                              spiox_out$B %>% tail(c(NA, NA, round(mcmc/2))), 
                              spiox_out$S %>% tail(c(NA, NA, round(mcmc/2))), 
                              spiox_out$theta %>% tail(c(NA, NA, round(mcmc/2))), 
                              num_threads = 16)

Ytest <- boom2$Y %>% apply(1:2, mean)
Ytrue <- Y[testset,]

1:q %>% sapply(\(j) cor(Ytest[,j], Ytrue[,j]))


spiox_out$theta %>% tail(c(NA,NA,round(mcmc/2))) %>% apply(1:2, mean)

Sig_est <- round(mcmc/2):mcmc %>% sapply(\(m) with(spiox_out,  diag(sqrt(theta[2,,m])) %*% 
                                         Sigma[,,m] %*% diag(sqrt(theta[2,,m])) ) ) %>% 
  array(c(q,q,mcmc)) %>% apply(1:2, mean)

cbind(spiox_out$theta %>% apply(1:2, mean), theta_true) %>% data.frame() %>% arrange(theta_true)

Cor_mcmc <- spiox_out$Sigma %>% tail(c(NA, NA, round(mcmc/2))) %>% apply(3, \(x) cov2cor(x)) %>%
  array(dim=c(q,q,round(mcmc/2)))

image(abs(cov2cor(Sigma) - apply(Cor_mcmc, 1:2, mean)))

# microergodics
diag(Sigma) * theta_true
j <- 2
plot(spiox_out$theta[j,] * 
       spiox_out$Sigma[j,j,], type='l')


df_test <- data.frame(cx[testset,], y=Wtest) %>%
  pivot_longer(cols=-c(Var1, Var2))
ggplot(df_test, aes(Var1, Var2, color=value)) +
  geom_point() +
  facet_grid(~name) +
  scale_color_viridis_c() +
  theme_minimal()






mesh_total_time <- system.time({
  meshout <- meshed::spmeshed(y=Y, 
                              x=X,
                              coords=cx, k = q,
                              block_size = 50, 
                              n_samples = 5000, 
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




set.seed(1)
(total_time2 <- system.time({
  spassso_out <- spassso::spassso(Y[-testset,], 
                                  X[-testset,,drop=F],
                                  cx[-testset,], 
                                  radgp_rho = radgp_rho, theta=theta_opts,
                                  
                                  spf_k = kfit, spf_a_delta = .1, spf_b_delta = .1, spf_a_dl = 0.5,
                                  
                                  spf_Lambda_start = Lambda[,1:kfit] %>% perturb(),
                                  spf_Delta_start = runif(q),#diag(Delta),
                                  mvreg_B_start = Beta,# %>% perturb(),
                                  
                                  mcmc = mcmc <- 1000,
                                  print_every=100,
                                  
                                  sample_precision=1,
                                  sample_mvr=T,
                                  sample_gp=T)
}))



spf_test <- spiox::run_spf_model(Y, kfit, .1, .1, 0.5, 
                                 Lambda[,1:kfit] %>% perturb(), runif(q), 
                                 mcmc = mcmc, 100, seq_lambda = F)



Qc_samples <- 1:mcmc %>% 
  sapply(\(i) with(spf_test, cov2cor(tcrossprod(Lambda[,,i]) + diag(Delta[,i]) ))) %>% 
  array(dim=c(q, q, mcmc))

Qc_post_spf <- Qc_samples %>% apply(1:2, mean)

mean((Qc_post_spf-cov2cor(Q))^2)



Q_est_pattern <- Q_samples %>% tail(c(NA,NA,round(3*mcmc/4))) %>% 
  apply(1:2, \(x) 1*(sign(quantile(x, 0.05))==sign(quantile(x, 0.95)))) 

Q_true_pattern <- 1*(Q!=0) 

# percentage of correct off-diagonal nonzeros 
check_pattern <- Q_true_pattern == Q_est_pattern
mean(check_pattern)
#(sum(check_pattern) - q)/(prod(dim(Q)) - q)


S_samples <- 1:mcmc %>% sapply(\(i) 
                               with(spiox_out, solve(tcrossprod(Lambda[,,i]) + diag(Delta[,i])) ) ) %>% 
  array(dim=c(qstar,qstar,mcmc))

B_reg <- 1:mcmc %>% sapply(\(i) {
  S <- with(spiox_out, solve(tcrossprod(Lambda[,,i]) + diag(Delta[,i])) )
  return( S[-(1:pstar),1:pstar] %*% solve(S[1:pstar,1:pstar]) )
}
) %>% 
  array(dim=c(q,pstar,mcmc))

image(B_reg %>% apply(1:2, mean))
image(Bstar)


B_true_pattern <- 1*(Bstar!=0)
B_est_pattern <- B_reg %>% tail(c(NA,NA,round(3*mcmc/4))) %>% 
  apply(1:2, \(x) 1*(sign(quantile(x, 0.05))==sign(quantile(x, 0.95)))) 

# percentage of correct nonzeros 
check_pattern_B <- B_est_pattern == B_true_pattern
mean(check_pattern_B)


## setting zeros


eps <- 0.05

chandra_fdr <- function(eps, Qc_samples, beta=0.9){
  H1 <- Qc_samples %>% apply(1:2, \(x) abs(x)>eps)
  H0 <- Qc_samples %>% apply(1:2, \(x) abs(x)<eps)
  
  post_H1 <- H1 %>% apply(2:3, mean)
  post_H0 <- H0 %>% apply(2:3, mean)
  
  dij <- 1*(post_H1>beta)
  
  fdr <- sum(dij*post_H0)/max(c(sum(dij), 1))
  
  return(fdr)
}

eps_grid <- seq(0.05, 0.7, length.out=100)

fdr_grid <- eps_grid %>% sapply(chandra_fdr, Qc_samples %>% tail(c(NA, NA, 1000)))

plot(eps_grid, fdr_grid, type='l')


Qc_est_pattern <- abs(Qc_samples %>% tail(c(NA,NA,round(1000))) %>% 
                        apply(1:2, mean)) > 0.3

Q_true_pattern <- 1*(Q!=0) 

image(Qc_est_pattern)
image(Q_true_pattern)
mean(Q_true_pattern == Qc_est_pattern)
















preds <- ROCR::prediction(Q_est_pattern[lower.tri(Q_est_pattern)], Q_true_pattern[lower.tri(Q_true_pattern)])
(perf <- ROCR::performance(preds, "fpr")@y.values[[1]][2])



