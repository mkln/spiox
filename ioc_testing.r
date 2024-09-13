rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0.05, .91, length.out=3)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)


# multivariate
q <- 2
Q <- rWishart(1, q+2, diag(q))[,,1] #
#tcrossprod(Lambda) + Delta

Sigma <- solve(Q) 


philist <- c(1,10,1.5)[1:q]
Clist <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1) ) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Lilist <- Llist %>% lapply(\(L) solve(L))


C <- Matrix::bdiag(Llist) %*% (Sigma %x% diag(nr)) %*% t(Matrix::bdiag(Llist))




new1 <- matrix(c(0.1, 0.1), ncol=2)
new2 <- matrix(c(10.35, 0.1), ncol=2)
colnames(new1) <- colnames(new2) <- c("Var1", "Var2")

cnew1 <- rbind(coords, new1)
cnew2 <- rbind(coords, new2)


C1_l <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cnew1))^1) ) )
L1_l <- C1_l %>% lapply(\(C) t(chol(C)))

C2_l <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cnew2))^1) ) )
L2_l <- C2_l %>% lapply(\(C) t(chol(C)))

Lbig <- list()
for(j in 1:q){
  Lbig[[j]] <- matrix(0, nr+2, nr+2)
  
  Lbig[[j]][1:nr, 1:nr] <- Llist[[j]]
  Lbig[[j]][nr+1, 1:(nr+1)] <- L1_l[[j]][nr+1,]
  Lbig[[j]][nr+2, c(1:nr, nr+2)] <- L2_l[[j]][nr+1,]
}

A <- matrix(rnorm(q*q), ncol=q)
A[upper.tri(A)] <- 0 
diag(A) <- abs(diag(A))

A[2,] <- c(0.1, 0.2)




xx <- seq(0.05, 2, length.out=15)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)


Sigma <- tcrossprod(A)

philist <- c(1,3)

Cmat <- \(l1, l2){
  matrix(c(
    Sigma[1,1] * ioc_xcor(l1, l2, 1, 1, cx, philist),
    Sigma[2,1] * ioc_xcor(l1, l2, 2, 1, cx, philist),
    Sigma[1,2] * ioc_xcor(l1, l2, 1, 2, cx, philist),
    Sigma[2,2] * ioc_xcor(l1, l2, 2, 2, cx, philist)
  ), ncol=2)
}

Cmat(cx[1,,drop=F], cx[nr,,drop=F])

Sigma[1,2] * ioc_xcor(new1, new2, 2, 1, cx, philist)/
  sqrt( Sigma[1,1] * ioc_xcor(new1, new2, 1, 1, cx, philist) *
          Sigma[2,2] * ioc_xcor(new1, new2, 2, 2, cx, philist))


Sigma[1,2] * ioc_xcor(cx[1,,drop=F], cx[nr,,drop=F], 1, 2, cx, philist)/
    sqrt( Sigma[1,1] * ioc_xcor(cx[1,,drop=F], cx[nr,,drop=F], 1, 1, cx, philist) *
            Sigma[2,2] * ioc_xcor(cx[1,,drop=F], cx[nr,,drop=F], 2, 2, cx, philist))


# LMC

a <- 0.1
b <- 0.5
A <- matrix(c(1, a, 0, b), ncol=2)
philist <- c(1, 2)
rho <- 1:q %>% lapply(\(j) \(h) exp(-philist[j] * sqrt(sum(h^2)))   )

Drho <- \(h) { 1:2 %>% sapply(\(i) rho[[i]](h)) }

cov2cor(A %*% diag(Drho(.51)) %*% t(A))


G <- function(h, a, b, phi1, phi2){
  a * exp(-phi1*h) / sqrt(
    exp(-phi1*h) * (a^2 * exp(-phi1*h) + b^2 * exp(-phi2*h))
  )
}

G(.51, A[2,1], A[2,2], philist[1], philist[2])




