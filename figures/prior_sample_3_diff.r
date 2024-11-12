rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)
library(latex2exp)

set.seed(2024)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

q <- 4

theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(15, 15, 15, 15) # phi
theta_mat[2,] <- c(1, 1, 1, 1) # sigmasq
theta_mat[3,] <- c(1, 14.9, 1.25, 1.5) # nu
theta_mat[4,] <- rep(1e-3, q)

sds <- c(1, 1, 1, 1)
Omega <- cbind(c(1,-.9,0.7, 0.8), c(-.9,1,-.5, -0.7), c(0.7,-.5,1, 0.8), c(0.8, -0.7, 0.8, 1))
Sigma <- diag(sds) %*% Omega %*% diag(sds)

St <- chol(Sigma)
S <- t(St)

# spatial
xx <- seq(0, 1, length.out=200)
coords <- expand.grid(xx, xx)
nr <- nrow(coords)
cx <- as.matrix(coords)
A <- matrix(runif(nr), ncol=1)

m <- 40

custom_dag <- dag_vecchia(cx, m)


# warping for the 3rd outcome
warpx <- spiox::daggp_build(cx, custom_dag, phi=1, sigmasq=.03, 
                         nu=1, tausq=0, matern=1, 16)$H %>% solve(rnorm(nr))
warpy <- spiox::daggp_build(cx, custom_dag, phi=1, sigmasq=.03, 
                            nu=1, tausq=0, matern=1, 16)$H %>% solve(rnorm(nr))
cx_warp <- cx + cbind(warpx, warpy)

cxlist <- list(cx, cx, cx_warp, cx)

cov_choice <- c(1, 2, 1, 1)
Hlist <- 1:q %>% sapply(\(j) {cat(j, "\n") 
                        spiox::daggp_build(cxlist[[j]], custom_dag, 
                                           phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                                           nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                                           matern=cov_choice[j], 16)$H
})

# warping for the 3rd outcome

# sill for the 4th outcome
wsdelta <- spiox::daggp_build(cx, custom_dag, phi=30, sigmasq=.5, 
                         nu=1.5, tausq=0, matern=1, 16)$H %>% solve(rnorm(nr))

V <- matrix(rnorm(nr * q), ncol=q) %*% St
V[,4] <- (wsdelta+1) * V[,4]

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
  mutate(name = 
           ifelse(name == "y.1", "Outcome 1", 
                  ifelse(name == "y.2", "Outcome 2",
                         ifelse(name == "y.3", "Outcome 3", "Outcome 4"))))

( p2 <- ggplot(df, 
               aes(Var1, Var2, fill=value)) +
    geom_raster() +
    scico::scale_fill_scico(palette="vik") + # bam, broc, cork, managua, vik
    facet_wrap(~name, ncol=2) +
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

ggsave("figures/prior_sample_3_diff2.pdf", plot=p2, width=7, height=6.7)
ggsave("figures/prior_sample_3_diff2.png", plot=p2, width=7, height=6.7)

