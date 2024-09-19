rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0, 1, length.out=200)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

q <- 3

theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(5, 20, 30)
theta_mat[2,] <- c(1, 1, 1)
theta_mat[3,] <- c(0.5, 1.2, 1.9)
theta_mat[4,] <- rep(1e-6, q)

j <- 1
radius <- 0.025
test <- spiox::radgp_build(cx, radius, 
                           phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                           nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                           matern=T, 16)
test$dag %>% sapply(\(x) prod(dim(x))) %>% summary()

Hlist <- 1:q %>% sapply(\(j)
                        spiox::radgp_build(cx, radius, 
                                           phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                                           nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                                           matern=T, 16)$H )
sds <- c(1, 1, 1)
Omega <- matrix(c(1,-.5,0.9,-.5,1,-.3,0.9,-.3,1), ncol=3)
Sigma <- diag(sds) %*% Omega %*% diag(sds)

St <- chol(Sigma)
S <- t(St)

V <- matrix(rnorm(nr * q), ncol=q) %*% St

Y <- V

save_which <- rep(0, q)

# make q spatial outcomes
which_process <- rep(0, q)
for(i in 1:q){
  cat(i, "\n")
  Y[,i] <- Matrix::solve(Hlist[[i]], V[,i], sparse=T)
}

df <- data.frame(coords, y=Y) %>% 
  pivot_longer(cols=-c(Var1, Var2)) %>%
  mutate(name = ifelse(name == "y.1", "Outcome 1", ifelse(name == "y.2", "Outcome 2", "Outcome 3")))

( p2 <- ggplot(df, 
       aes(Var1, Var2, fill=value)) +
  geom_raster() +
  scico::scale_fill_scico(palette="vik") + # bam, broc, cork, managua, vik
  facet_wrap(~name, ncol=q) +
    theme_minimal() + 
    labs(x=NULL, y=NULL) +
    scale_x_continuous(breaks=c(0.5, 1), expand=c(0,0)) +
    scale_y_continuous(breaks=c(0, 0.5, 1), expand=c(0,0)) +
    theme(
      panel.grid = element_blank(),
      panel.spacing.x = unit(10, "pt"),
      panel.border = element_rect(fill=NA, color="black"),
      axis.text.x = element_text(margin = margin(t = 0), hjust=1),
      
      axis.text.y = element_text(margin = margin(r = 0), vjust=1) ) )

#ggsave("figures/prior_sample_3.pdf", plot=p2, width=8.7, height=3)



## compute correlation functions
# we only want to go linearly down testx and testy making only those comparisons, hence diag_only=T

library(spiox)

xx <- seq(0, 1, length.out=40)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)

xg <- seq(0.2, 0.8, length.out=5)#
  #xx[ seq(10, length(xx), 20) ]

iox_spcor <- function(outcome_i, outcome_j, h_values, 
                      x_test_grid, 
                      cx, theta_mat,
                      n_angles=5, diag_only=T, at_limit=T){
  
  all_combos_setup <- expand.grid(x_test_grid, x_test_grid, 
                                  h_values, seq(0,2*pi,length.out=n_angles)) %>% as.matrix()
  colnames(all_combos_setup) <- c("s1x", "s1y", "h", "angle")
  
  all_combos_setup %<>% data.frame() %>% 
    mutate(s2x = s1x + h*cos(angle), s2y = s1y + h*sin(angle))
  
  testx <- all_combos_setup %>% dplyr::select(s1x, s1y) %>% as.matrix()
  testy <- all_combos_setup %>% dplyr::select(s2x, s2y) %>% as.matrix()

  cov_computed <- iox(testx, testy, outcome_i, outcome_j, 
                      cx, theta_mat,
                      matern = TRUE,
                      diag_only=diag_only, limit=at_limit)

  all_combos_setup %<>% mutate(spcov = cov_computed)
  
  spcov_by_h <- all_combos_setup %>% group_by(h) %>% summarise(spcov = mean(spcov))
  
  return( spcov_by_h )
}

# limit variances at 0
marg_cor_lims <- 1:q %>% 
  lapply(\(j) iox_spcor(j, j, c(0), xg, cx, theta_mat, n_angles=8, diag_only=T, at_limit=T) %>% 
           mutate(outcome=j) ) %>%
  bind_rows() %>% 
  rename(cov0 = spcov)

hvec <- c(seq(0, 0.05, length.out=10), seq(0.06, .5, length.out=20))

combis <- cbind(c(1,1), c(2,2), c(3,3), c(1,2), c(1,3), c(2,3))

cor_all_combs <- plyr::alply(combis, 2, \(x) {
  i <- x[1]
  j <- x[2]
  
  spcov_by_h <- iox_spcor(i, j, hvec, xg, cx, theta_mat, 8) %>% 
    mutate(outcome_1 = i, outcome_2 = j) %>% 
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_1"="outcome")) %>%
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_2"="outcome")) %>%
    mutate(spcor = Sigma[i,j]/sqrt(Sigma[i,i]*Sigma[j,j]) * spcov/sqrt(cov0.x * cov0.y))
  return(spcov_by_h)
}) %>% bind_rows()


ggplot(cor_all_combs %>% dplyr::filter(outcome_1 == 1, outcome_2 == 1), aes(h, spcor)) +
  geom_line() +
  theme_minimal() +
  scale_y_continuous(limits=c(-1,1))

# correlation functions used to generate IOX
rho_corrs <- 1:q %>% lapply(\(i) {
  thetai <- theta_mat[,i]
  thetai[4] <- 0
  rho_corrs <- 
    hvec %>% sapply(\(h) spiox::Correlationc(matrix(c(0,0), nrow=1), matrix(c(h, 0), nrow=1), 
                                             theta = thetai, covar = T, same=F) )
  return(data.frame(h=hvec, outcome = i, spcor = rho_corrs))
}) %>% bind_rows()






################ WIP 


make_plot <- function(i, j){
  
  
  made_plot <- ggplot(spcov_by_h, aes(h, spcor)) +
    geom_line() +
    theme_minimal() +
    scale_y_continuous(limits=c(-1,1))
  
  if(i==j){
    
    made_plot <- made_plot + 
      geom_line(data=data.frame(h=hvec, spcor=rho_corrs), color="orange", lty="dashed") +
      scale_y_continuous(limits=c(0,1))
  }
  return(made_plot)
}


p11 <- make_plot(1, 1)
p22 <- make_plot(2, 2)
p33 <- make_plot(3, 3)

p12 <- make_plot(1, 2)
p13 <- make_plot(1, 3)
p23 <- make_plot(2, 3)

library(gridExtra)
library(grid)

grid.arrange(
  p11, p22, p33,
  p12, p13, p23, nrow=2)

i <- 1; j <- 1
spcov1_by_h <- iox_spcor(i, j, hvec, xx, cx, theta_mat, 8) %>% 
  mutate(outcome_1 = i, outcome_2 = j) %>% 
  left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_1"="outcome")) %>%
  left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_2"="outcome")) %>%
  mutate(spcor = Sigma[i,j]/sqrt(Sigma[i,i]*Sigma[j,j]) * spcov/sqrt(cov0.x * cov0.y))



spcov1_by_h %>% ggplot(aes(x=h, y=spcor)) +
  geom_line() +
  




