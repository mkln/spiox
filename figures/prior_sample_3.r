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
xx <- seq(0, 1, length.out=20)
coords <- expand.grid(xx, xx)
nr <- nrow(coords)
cx <- as.matrix(coords)
A <- matrix(runif(nr), ncol=1)

m <- 20

custom_dag <- dag_vecchia(cx, m)

test <- daggp_build_mm(A, cx, custom_dag,
               20, 1, 1, 1e-7, T)

Hlist <- 1:q %>% sapply(\(j)
                        spiox::daggp_build(cx, custom_dag, 
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
    labs(x=NULL, y=NULL, fill="Value") +
    scale_x_continuous(breaks=c(0.5, 1), expand=c(0,0)) +
    scale_y_continuous(breaks=c(0, 0.5, 1), expand=c(0,0)) +
    theme(
      panel.grid = element_blank(),
      panel.spacing.x = unit(10, "pt"),
      panel.border = element_rect(fill=NA, color="black"),
      axis.text.x = element_text(margin = margin(t = 0), hjust=1),
      
      axis.text.y = element_text(margin = margin(r = 0), vjust=1) ) )

