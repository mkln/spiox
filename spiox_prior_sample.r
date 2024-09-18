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

q <- 3

theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(5, 10, 10)
theta_mat[2,] <- c(1, 1, 1)
theta_mat[3,] <- c(0.5, .5, .5)
theta_mat[4,] <- rep(1e-6, q)


Hlist <- 1:q %>% sapply(\(j)
              spiox::radgp_build(cx, 0.1, 
                                 phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                                 nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                                 matern=T, 16)$H )

Sigma <- matrix(c(1,-.5,0.9,-.5,1,-.3,0.9,-.3,1), ncol=3) 

St <- chol(Sigma)
S <- t(St)

V <- matrix(rnorm(nr * q), ncol=q) %*% St

Y <- V

save_which <- rep(0, q)

# make q spatial outcomes
which_process <- rep(0, q)
for(i in 1:q){
  Y[,i] <- solve(Hlist[[i]], V[,i])
}

cov_Y_cols <- cov(Y) %>% as("dgCMatrix")
cov_Y_rows <- cov(t(Y)) %>% as("dgCMatrix")

df <- data.frame(coords, y=Y) %>% 
  pivot_longer(cols=-c(Var1, Var2))

p2 <- ggplot(df, 
       aes(Var1, Var2, fill=value)) +
  geom_raster() +
  scale_fill_viridis_c() +
  facet_wrap(~name, ncol=5)


hvec <- c(seq(1e-10, 0.3, length.out=20), seq(0, .3, length.out=15))
xg <- seq(0.2, 0.8, length.out=2)

all_combos_setup <- expand.grid(xg,xg,hvec,seq(0,2*pi,5)) %>% as.matrix()
colnames(all_combos_setup) <- c("s1x", "s1y", "h", "angle")

all_combos_setup %<>% data.frame() %>% 
  mutate(s2x = s1x + h*cos(angle), s2y = s1y + h*sin(angle))

testx <- all_combos_setup %>% dplyr::select(s1x, s1y) %>% as.matrix()
testy <- all_combos_setup %>% dplyr::select(s2x, s2y) %>% as.matrix()

# we only want to go linearly down testx and testy making only those comparisons, hence diag_only=T
test <- iox(testx, testy, 3, 3, cx, theta_mat, diag_only=T)

all_combos_setup %<>% mutate(spcov = test)

spcov_by_h <- all_combos_setup %>% group_by(h) %>% summarise(spcov = mean(spcov))
#at_zero <- spcov_by_h %>% dplyr::filter(h==0) %$% spcov
#spcov_by_h %<>% mutate(corr = spcov/at_zero)

ggplot(spcov_by_h, aes(h, spcov)) +
  geom_line()












