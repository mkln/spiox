
rm(list=ls())
library(tidyverse)
library(magrittr)

library(Matrix)
image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}


set.seed(4)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}
sumabs <- \(x) { 
  sum(abs(x))
}

# spatial
xx <- seq(0, 1, length.out=30)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

k <- 5
philist <- c(2.5, 6.1, 0.5, 5.1, 0.1)[1:k]
explist <- c(1, 1.5, 1.8, 1, 1.5)[1:k]
Clist <- 1:k %>% lapply(\(i)  (exp(-philist[i] * as.matrix(dist(cx))^explist[i]) ) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))

#Q <- matrix(c(2,1,0,1,2,1,0,1,2), ncol=3)
#Sigma <- solve(Q)
Sigma <- rWishart(1, 10, diag(k))[,,1] %>% cov2cor()
#Sigma <- matrix(c(2,1.8,1.8,2), ncol=2)
#Sigma <- diag(k)
A <- t(chol(Sigma))

I <- \(x) diag(x)

Lblocks <- Matrix::bdiag(Llist)

Cblocks <- Lblocks %*% (Sigma %x% I(nr)) %*% t(Lblocks)

#Ciblocks <- solve(t(Lblocks)) %*% (solve(Sigma) %x% I(nr)) %*% solve(Lblocks)
#C_LMC <- (A %x% I(nr)) %*% Matrix::bdiag(Clist) %*% (t(A) %x% I(nr))
#Ci_LMC <- solve(C_LMC)

v <- rnorm(nr*k)
y <- Lblocks %*% (A %x% I(nr)) %*% v
w <- (A %x% I(nr)) %*% Lblocks %*% v


Y <- matrix(y, ncol=k)
W <- matrix(w, ncol=k)

df <- data.frame(coords, y=W) %>%
  pivot_longer(cols=c(-Var1, -Var2))

ggplot(df, aes(Var1, Var2, fill=value)) +
  geom_raster() +
  scale_fill_viridis_c() +
  facet_grid(~name)


newc <- c(0.124, 0.531)
Cnlist <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(rbind(cx,newc)))^1) ) )
Lnlist <- Cnlist %>% lapply(\(C) t(chol(C)))
Lnblocks <- Matrix::bdiag(Lnlist)


S <- solve(Q)
B <- t(chol(S))

C_big <- Lnblocks %*% (S %x% I(nr+1)) %*% t(Lnblocks)
#C_big <- (B %x% I(nr+1)) %*% Matrix::bdiag(Cnlist) %*% (t(B) %x% I(nr+1))
#C_big <- S %x% Cnlist[[1]]

outix <- c(26, 52, 78)

H <- C_big[outix, -outix] %*% solve(C_big[-outix, -outix]) 
R <- C_big[outix, outix] - H %*% C_big[-outix, outix]

H %*% as.vector(y)

ych <- y
#ych[(nr+1):(k*nr)] <- ych[(nr+1):(k*nr)]*2
#H %*% as.vector(ych)




Lnlist[[1]]
solve(Llist[[1]], Cnlist[[1]][1:nr, nr+1])


Hcheck <- Matrix::bdiag(Lnlist %>% lapply(\(L) head(tail(L, 1), c(NA,-1)))) %*% 
  (Sigma %x% I(nr)) %*%
  #Matrix::bdiag(Lnlist %>% lapply(\(L) t(head(L, c(nr, nr))))) %*% Ciblocks
  Matrix::bdiag(Llist %>% lapply(\(L) t(L))) %*% Ciblocks

Hcheck %*% as.vector(y)


Cnewold <- matrix(tail(Cnlist[[1]],1)[1:nr], ncol=1)

- Cnewold %*% solve(Clist[[1]]) * 
as.numeric(sqrt(1 - Cnewold %*% solve(Clist[[1]]) %*% t(Cnewold)))




outH <- function(C){
  return(
    -C[nr+1,1:nr] %*% solve(C[1:nr,1:nr]) * (1/sqrt(C[nr+1, nr+1] - C[nr+1,1:nr] %*% solve(C[1:nr,1:nr]) %*% C[1:nr, nr+1])[1,1]) 
  )
}

C <- Cnlist[[1]]
ImH <- c(-C[nr+1,1:nr] %*% solve(C[1:nr,1:nr]), 1)
Rihalf <- 1/sqrt(C[nr+1, nr+1] - C[nr+1,1:nr] %*% solve(C[1:nr,1:nr]) %*% C[1:nr, nr+1])[1,1]


L1 <- outH(Cnlist[[1]])
L2 <- outH(Cnlist[[2]])
L3 <- outH(Cnlist[[3]])
Matrix::bdiag(L1, L2, L3)






