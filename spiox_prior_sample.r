rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(latex2exp)

set.seed(4)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

q <- 3

theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(5, 15, 30)
theta_mat[2,] <- c(1, 1, 1)
theta_mat[3,] <- c(0.5, 1.2, 1.9)
theta_mat[4,] <- rep(1e-3, q)

sds <- c(1, 1, 1)
Omega <- matrix(c(1,-.9,0.7,-.9,1,-.5,0.7,-.5,1), ncol=3)
Sigma <- diag(sds) %*% Omega %*% diag(sds)

St <- chol(Sigma)
S <- t(St)

# spatial
xx <- seq(0, 1, length.out=200)
coords <- expand.grid(xx, xx)
nr <- nrow(coords)
cx <- as.matrix(coords)

j <- 1
radius <- 0.025
test <- spiox::radgp_build(cx, radius, 
                           phi=5, sigmasq=1, 
                           nu=1.5, tausq=1e-5, 
                           matern=F, 16)
test$dag %>% sapply(\(x) prod(dim(x))) %>% summary()

Hlist <- 1:q %>% sapply(\(j)
                        spiox::radgp_build(cx, radius, 
                                             phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                                             nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                                             matern=T, 16)$H
                         )
                        

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
  
ggsave("figures/prior_sample_3.pdf", plot=p2, width=8.7, height=3)
  




## compute correlation functions
# we only want to go linearly down testx and testy making only those comparisons, hence diag_only=T

library(spiox)
set.seed(1)
xx <- seq(0, 1, length.out=10)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
n_g <- 4
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
                      diag_only=diag_only, at_limit=at_limit)
  
  all_combos_setup %<>% mutate(spcov = cov_computed)
  
  spcov_by_h <- all_combos_setup %>% group_by(h) %>% summarise(spcov = mean(spcov))
  
  return( spcov_by_h )
}


xg_base <- xx[ seq(1, length(xx), length(xx)/(n_g+1)) ] %>% tail(-1)


## compute IOX covariance between two locations both in S^c
xg <- xg_base + runif(n_g, 0, 1e-2)
# limit variances at 0
marg_cor_lims <- 1:q %>% 
  lapply(\(j) iox_spcor(j, j, c(0), xg, cx, theta_mat, n_angles=8, diag_only=T, at_limit=T) %>% 
           mutate(outcome=j) ) %>%
  bind_rows() %>% 
  rename(cov0 = spcov)

hvec <- c(10^seq(-5, -1, length.out=25), 10^seq(-1, -0.5, length.out=25))
  #c(seq(0, 0.05, length.out=10), seq(0.06, .5, length.out=20))

combis <- cbind(c(1,1), c(2,2), c(3,3), c(1,2), c(1,3), c(2,3))

cor_all_combs_Sc <- plyr::alply(combis, 2, \(x) {
  i <- x[1]
  j <- x[2]
  
  spcov_by_h <- iox_spcor(i, j, hvec, xg, cx, theta_mat, 10) %>% 
    mutate(outcome_1 = i, outcome_2 = j) %>% 
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_1"="outcome")) %>%
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_2"="outcome")) %>%
    mutate(spcor = #Sigma[i,j]/sqrt(Sigma[i,i]*Sigma[j,j]) * spcov/sqrt(cov0.x * cov0.y))
                spcov)
  return(spcov_by_h)
}) %>% 
  bind_rows() %>%
  mutate(model="IOX_Sc")

## compute IOX covariance between two locations, one in S
xg <- xg_base 
# limit variances at 0
marg_cor_lims <- 1:q %>% 
  lapply(\(j) iox_spcor(j, j, c(0), xg, cx, theta_mat, n_angles=8, diag_only=T, at_limit=T) %>% 
           mutate(outcome=j) ) %>%
  bind_rows() %>% 
  rename(cov0 = spcov)

hvec <- c(10^seq(-5, -1, length.out=25), 10^seq(-1, -0.5, length.out=25))
#c(seq(0, 0.05, length.out=10), seq(0.06, .5, length.out=20))

combis <- cbind(c(1,1), c(2,2), c(3,3), c(1,2), c(1,3), c(2,3))

cor_all_combs_S <- plyr::alply(combis, 2, \(x) {
  i <- x[1]
  j <- x[2]
  
  spcov_by_h <- iox_spcor(i, j, hvec, xg, cx, theta_mat, 10) %>% 
    mutate(outcome_1 = i, outcome_2 = j) %>% 
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_1"="outcome")) %>%
    left_join(marg_cor_lims %>% dplyr::select(-h), by=c("outcome_2"="outcome")) %>%
    mutate(spcor = Sigma[i,j] * #/sqrt(Sigma[i,i]*Sigma[j,j]) * spcov/sqrt(cov0.x * cov0.y))
              spcov)
  return(spcov_by_h)
}) %>% 
  bind_rows() %>%
  mutate(model="IOX_S")

cor_all_combs <- cor_all_combs_S #bind_rows(cor_all_combs_Sc, cor_all_combs_S)

ggplot(cor_all_combs %>% dplyr::filter(outcome_1 == 1, outcome_2 == 2), aes(h, spcor)) +
  geom_line(aes(color=model)) +
  theme_minimal() +
  scale_y_continuous(limits=c(-1,1))


# correlation functions used to generate IOX
rho_corrs <- 1:q %>% lapply(\(i) {
  thetai <- theta_mat[,i]
  thetai[4] <- 0
  rho_corrs <- 
    hvec %>% sapply(\(h) spiox::Correlationc(matrix(c(0,0), nrow=1), matrix(c(h, 0), nrow=1), 
                                             theta = thetai, covar = T, same=F) )
  return(data.frame(h=hvec, outcome_1 = i, outcome_2 = i, spcor = rho_corrs, model="Matern"))
}) %>% bind_rows()


corr_plot <- cor_all_combs %>% bind_rows(rho_corrs)

corr_plot$model <- factor(corr_plot$model)
levels(corr_plot$model) <- c(TeX("IOX $l \\in S$"), TeX("IOX $l,l' \\in S^c$"), "Matern" )





################ WIP 


cross_plots <- ggplot(corr_plot %>% dplyr::filter(grepl("IOX", model), outcome_1 != outcome_2), 
                    aes(h, spcor)) +
  geom_line(aes(color=model)) +
  theme_minimal() + 
  scale_color_manual(values=c("black", "red"), labels = scales::parse_format()) +
  facet_wrap(outcome_1 ~ outcome_2) +
  scale_y_continuous(limits=c(-1,1))

marg_plots <- ggplot(corr_plot %>% dplyr::filter(outcome_1 == outcome_2), 
                     aes(h, spcor)) +
  geom_line(aes(color=model)) +
  theme_minimal() + 
  scale_color_manual(values=c("black", "red", "blue"), labels = scales::parse_format()) +
  facet_wrap(outcome_1 ~ outcome_2) +
  scale_y_continuous(limits=c(0,1))




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
  




