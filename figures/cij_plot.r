rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(latex2exp)
library(scico)

set.seed(4)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

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


q <- 2

cor_pair <- function(phi, nu){
  
  theta_mat <- matrix(1, ncol=2, nrow=4)
  theta_mat[1,] <- c(10, phi)
  theta_mat[2,] <- c(1, 1)
  theta_mat[3,] <- c(1, nu)
  theta_mat[4,] <- rep(1e-10, 2)
  
  set.seed(1)
  xx <- seq(0, 1, length.out=10)
  coords <- expand.grid(xx, xx)
  cx <- as.matrix(coords)
  n_g <- 5
  
  xg <- xx[ seq(1, length(xx), length(xx)/(n_g+1)) ] %>% tail(-1)
  hvec <- c(10^seq(-5, -2, length.out=25), 10^seq(-2, -0.5, length.out=25))
  
  spcor <- iox_spcor(1, 2, hvec, xg, cx, theta_mat, n_angles=10, diag_only=T, at_limit=F) 
  return(spcor %>% mutate(phi = phi, nu = nu))
}

phinu <- expand.grid(phi = seq(1, 40, 2),
                     nu = 1) %>% bind_rows(
                       expand.grid(phi = 10, nu = seq(0.5, 1.9, length.out=25)))

results <- phinu %>% as.matrix() %>% plyr::alply(1, \(x) cor_pair(x[1], x[2]))

phi_corr <- results %>% bind_rows() %>% 
  filter(nu==1) %>% 
  ggplot(aes(h, spcov, group=phi, color=phi)) + 
  scale_color_scico(palette="managua") +
  geom_line()  +
  theme_minimal() +
  labs(color=TeX("$\\phi$"), x="Distance", y="Correlation")

nu_corr <- results %>% bind_rows() %>% 
  filter(phi==10) %>% 
  ggplot(aes(h, spcov, group=nu, color=nu)) + 
  scale_color_scico(palette="managua", direction = -1) +
  geom_line() +
  theme_minimal() +
  labs(color=TeX("$\\nu$"), x="Distance", y="Correlation")

cij_plot <- grid.arrange(phi_corr, nu_corr, nrow=1)
ggsave("figures/cij_plot.pdf", plot=cij_plot, width=8.75, height=3)
ggsave("figures/cij_plot.png", plot=cij_plot, width=8.75, height=3)
