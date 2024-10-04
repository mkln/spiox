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
xS <- seq(0, 1, length.out=50)
cooS <- expand.grid(xS, xS)
nrS <- nrow(cooS)
cxS <- as.matrix(cooS) + matrix(runif(nrS*2, -1e-5, 1e-5), ncol=2)

xU <- seq(0, 1, length.out=200)
cooU <- expand.grid(xU, xU)
cxU <- as.matrix(cooU)

coords <- bind_rows(cooS %>% mutate(type="S"), 
                    cooU %>% mutate(type="U"))

m <- 50

custom_dag_S <- dag_vecchia(cxS, m)
custom_dag_U <- dag_vecchia_predict(cxS, cxU, m = m)

Hlist <- 1:q %>% sapply(\(j)
                        spiox::daggp_build(rbind(cxS, cxU), c(custom_dag_S, custom_dag_U), 
                                           phi=theta_mat[1,j], sigmasq=theta_mat[2,j], 
                                           nu=theta_mat[3,j], tausq=theta_mat[4,j], 
                                           matern=T, 16)$H
)

nr <- nrow(cooS) + nrow(cooU)
  
V <- matrix(rnorm(nr * q), ncol=q) %*% St

Y <- V

save_which <- rep(0, q)

# make q spatial outcomes
which_process <- rep(0, q)
for(i in 1:q){
  cat(i, "\n")
  Y[,i] <- Matrix::solve(Hlist[[i]], V[,i], sparse=T) 
}

df <- data.frame(coords, y=Y) %>% filter(type=="U") %>% 
  pivot_longer(cols=-c(Var1, Var2, type)) %>%
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

ggsave("figures/prior_sample_Sc.pdf", plot=p2, width=8.7, height=3)
