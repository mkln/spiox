rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0, 1, length.out=45)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

philist <- c(1,2.5,5.0)
Clist <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1) + 1e-15*diag(nr)) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Lilist <- Llist %>% lapply(\(L) solve(L))

# multivariate
q <- 6
k <- 2#round(q/2)

Lambda <- matrix(runif(q*k), ncol=k)
threshold_val <- .9
threshold <- abs(Lambda) < threshold_val
Lambda[threshold] <- 0
Lambda[!threshold] <- 5*rnorm(sum(!threshold))

Delta <- diag(runif(q, 0.1, .4))

Q <- rWishart(1, q+2, diag(q))[,,1] #
#tcrossprod(Lambda) + Delta

# percentage of off-diagonal nonzeros 
(sum(Q!=0) - q)/(prod(dim(Q)) - q)

Sigma <- solve(Q) 

St <- chol(Sigma)
S <- t(St)

V <- matrix(rnorm(nr * q), ncol=q) %*% St

Y_sp <- V

save_which <- rep(0, q)

# make q spatial outcomes
which_process <- rep(0, q)
for(i in 1:q){
  which_process[i] <- 1:length(philist) %>% sample(1)
  Y_sp[,i] <- Llist[[which_process[i]]] %*% V[,i]
}
theta_true <- philist[which_process]

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
kfit <- k
## Sp2sp
set.seed(1)

radgp_rho <- 0.1
test_radgp <- inocs::radgp_build(cx, radgp_rho, phi=1, sigmasq=1, nu=1, tausq=0, matern=F)
test_radgp$dag %>% sapply(\(x) nrow(x)) %>% summary()

  phi_opts <- seq(0.5, 10, length.out=3)

theta_opts <- phi_opts %>% sapply(\(phi) matrix( c(phi, 1, 1, 1e-9), ncol=1))

testset <- sample(1:nrow(Y), 10, replace=F)

perturb <- function(x, sd=1){
  return(x + matrix(rnorm(prod(dim(x)), sd), ncol=ncol(x)))
}

set.seed(1) 
(total_time <- system.time({
  inocs_out <- inocs::inocs(Y[-testset,,drop=F], 
                            X[-testset,,drop=F],
                            cx[-testset,], 
                            radgp_rho = radgp_rho, theta=theta_opts,
                            
                            spf_k = kfit, spf_a_delta = .1, spf_b_delta = .1, spf_a_dl = 0.5,
                            
                            spf_Lambda_start = Lambda[,1:kfit,drop=F] %>% perturb(),
                            spf_Delta_start = matrix(runif(q),ncol=1),#diag(Delta),
                            mvreg_B_start = Beta,# %>% perturb(),
                            
                            mcmc = mcmc <- 1000,
                            print_every=200,
                            
                            sample_precision=1,
                            sample_mvr=T,
                            sample_gp=T)
}))
cbind(inocs_out$theta %>% apply(1, mean), theta_true)

obj_out <- inocs_out

#sparse test
sLambda <- obj_out$Lambda
#sLambda[abs(sLambda) < quantile(abs(sLambda), 0.95)] <- 0

Sig_samples <- 1:mcmc %>% 
  sapply(\(i) with(obj_out, solve(tcrossprod( sLambda[,,i] ) + diag(Delta[,i]) ))) %>% 
  array(dim=c(q, q, mcmc))

# microergodics
diag(Sigma) * theta_true
j <- 6
plot(inocs_out$theta[1,j,] * 
       Sig_samples[j,j,], type='l')


Qa_samples <- 1:mcmc %>% 
  sapply(\(i) with(obj_out, cov2cor(tcrossprod( sLambda[,,i] ) + diag(Delta[,i]) ))) %>% 
  array(dim=c(q, q, mcmc))

Qa_post_sjns <- Qa_samples %>% apply(1:2, mean)


mean((Qa_post_sjns-cov2cor(Q))^2)


image(1*(Qa_post_sjns!=0))
mean(Q_true_pattern == (Qa_post_sjns!=0))


Q_samples <- 1:mcmc %>% 
  sapply(\(i) with(inocs_out, tcrossprod( Si[,,i] ) )) %>% 
  array(dim=c(q, q, mcmc))

Q_post_sjns <- Q_samples %>% apply(1:2, mean)


mean((Q_post_sjns-Q)^2)


Qc_samples <- 1:mcmc %>% 
  sapply(\(i) with(inocs_out, cov2cor( tcrossprod( Si[,,i] )))) %>% 
  array(dim=c(q, q, mcmc))

Qc_post_sjns <- Qc_samples %>% apply(1:2, mean)

mean((Qc_post_sjns - cov2cor(Q))^2)

#Q_samples[,,2500]

Qc_est_pattern <- Qc_samples %>% tail(c(NA,NA,round(3*mcmc/4))) %>% 
  apply(1:2, \(x) 1*(sign(quantile(x, 0.025))==sign(quantile(x, 0.975)))) 

Q_true_pattern <- 1*(Q!=0) 

image(Qc_est_pattern)
image(Q_true_pattern)
mean(Q_true_pattern == Qc_est_pattern)

Q_est_pattern <- Q_samples %>% tail(c(NA,NA,round(3*mcmc/4))) %>% 
  apply(1:2, \(x) 1*(sign(quantile(x, 0.005))==sign(quantile(x, 0.995)))) 

# percentage of correct off-diagonal nonzeros 

mean(Q_true_pattern == Q_est_pattern)


#(sum(check_pattern) - q)/(prod(dim(Q)) - q)
image(Q_est_pattern)




# alternative models



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



