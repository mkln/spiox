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

Ctest <- inocs::Correlationc(cx, cx, c(1,1,1.9,1-8), 1, TRUE)

q <- 15

optlist <- seq(0.4, 1.9, length.out=q) %>% sample(q, replace=T)

#Clist <- optlist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1) + 1e-6*diag(nr)) )
Clist <- optlist %>% lapply(\(nu) inocs::Correlationc(cx, cx, c(20,1,nu,1e-8), 1, TRUE) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Lilist <- Llist %>% lapply(\(L) solve(L))


Q <- rWishart(1, q+2, diag(q))[,,1] #
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

radgp_rho <- .12
test_radgp <- inocs::radgp_build(cx, radgp_rho, phi=1, sigmasq=1, nu=1, tausq=0, matern=F)
test_radgp$dag %>% sapply(\(x) nrow(x)) %>% summary()

theta_opts <- seq(0.6, 1.8, length.out=5)

theta_opts <- theta_opts %>% sapply(\(nu) matrix( c(20, 1, nu, 1e-6), ncol=1))

testset <- sample(1:nrow(Y), 10, replace=F)

perturb <- function(x, sd=1){
  return(x + matrix(rnorm(prod(dim(x)), sd), ncol=ncol(x)))
}

RhpcBLASctl::blas_set_num_threads(16)
RhpcBLASctl::omp_set_num_threads(16)

set.seed(1) 
(total_time <- system.time({
  inocs_out <- inocs::inocs_wishart(Y[-testset,,drop=F], 
                            X[-testset,,drop=F],
                            cx[-testset,], 
                            radgp_rho = radgp_rho, theta=theta_opts,
                            
                            Sigma_start = diag(q),
                            mvreg_B_start = 0*Beta,# %>% perturb(),
                            
                            mcmc = mcmc <- 1000,
                            print_every = 50,
                            
                            sample_iwish=T,
                            sample_mvr=T,
                            sample_gp=T)
}))


set.seed(1) 
(total1_time <- system.time({
  inocs1_out <- inocs::inocs_wishart(Y[-testset,1,drop=F], 
                                    X[-testset,,drop=F],
                                    cx[-testset,], 
                                    radgp_rho = radgp_rho, theta=theta_opts[,1,drop=F],
                                    
                                    Sigma_start = diag(1),
                                    mvreg_B_start = 0*Beta[,1,drop=F],# %>% perturb(),
                                    
                                    mcmc = mcmc <- 10000,
                                    print_every = 50,
                                    
                                    sample_iwish=T,
                                    sample_mvr=T,
                                    sample_gp=T)
}))






cbind(inocs_out$theta %>% apply(1, mean), theta_true)

inocs_out$Sigma %>% tail(c(NA, NA, round(mcmc/2))) %>% apply(1:2, mean)

# microergodics
diag(Sigma) * theta_true
j <- 2
plot(inocs_out$theta[j,] * 
       inocs_out$Sigma[j,j,], type='l')






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



spf_test <- inocs::run_spf_model(Y, kfit, .1, .1, 0.5, 
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
                               with(inocs_out, solve(tcrossprod(Lambda[,,i]) + diag(Delta[,i])) ) ) %>% 
  array(dim=c(qstar,qstar,mcmc))

B_reg <- 1:mcmc %>% sapply(\(i) {
  S <- with(inocs_out, solve(tcrossprod(Lambda[,,i]) + diag(Delta[,i])) )
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



boom <- inocs::inocs_predict(coords_new = cx[testset,],
                             X_new = X[testset,,drop=F],
                             Xstar_new = Xstar[testset,,drop=F],
                             
                             # data
                             Y[-testset,], 
                             X[-testset,,drop=F], 
                             Xstar[-testset,,drop=F], 
                             cx[-testset,], 
                             
                             radgp_rho, theta_opts, 
                             inocs_out$B %>% tail(c(NA, NA, 100)), 
                             inocs_out$S %>% tail(c(NA, NA, 100)), 
                             inocs_out$theta_which %>% tail(c(NA, 100)))

Wtest <- boom$Y %>% apply(1:2, mean)
Wtrue <- XY[testset,]

df_test <- data.frame(cx[testset,], y=Wtest) %>%
  pivot_longer(cols=-c(Var1, Var2))
ggplot(df_test, aes(Var1, Var2, color=value)) +
  geom_point() +
  facet_grid(~name) +
  scale_color_viridis_c() +
  theme_minimal()



