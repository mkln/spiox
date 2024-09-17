rm(list=ls())
library(tidyverse)
library(magrittr)
library(Matrix)

set.seed(2)

sumabs <- \(x) sum(abs(x))
trroot <- \(x) sum( sqrt(svd(x)$d ))
svdroot <- \(x) with(svd(x), u %*% diag(sqrt(d)) %*% t(u))
image.matrix <- \(x) {
  image(as(x, "dgCMatrix"))
}

# spatial
xx <- seq(0, 1, length.out=5)
coords <- #cbind(runif(100), runif(100))#
  expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)


# multivariate
q <- 2
Q <- rWishart(1, q+2, diag(q))[,,1] #
#tcrossprod(Lambda) + Delta

Sigma <- solve(Q) 

philist <- c(3, 1, 1.5)[1:q]

Clist <- philist %>% lapply(\(phi)  (exp(-phi * as.matrix(dist(cx))^1.9) ) )
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Llist2 <- Clist %>% lapply(\(C) with(svd(C), u %*% diag(sqrt(d))) )

Croots <- Clist %>% lapply(\(C) with(svd(C), u %*% diag(sqrt(d)) %*% t(u) ) )

Matrix::trac

Lilist <- Llist %>% lapply(\(L) solve(L))

Cbig <- Matrix::bdiag(Llist) %*% (Sigma %x% diag(nr)) %*% t(Matrix::bdiag(Llist))
Cbig2 <- Matrix::bdiag(Llist2) %*% (Sigma %x% diag(nr)) %*% t(Matrix::bdiag(Llist2))
Cbig3 <- Matrix::bdiag(Croots) %*% (Sigma %x% diag(nr)) %*% t(Matrix::bdiag(Croots))

hvec <- seq(0, .5, length.out=25)
xg <- seq(0.2, 0.8, length.out=2)
test_coords <- expand.grid(xg,xg) %>% as.matrix()

testh <- iox_cross_avg(hvec, 2, 1,
                       test_coords, cx, philist, 10, 1)
plot(hvec, testh, type='l', ylim=c(0,1))
lines(hvec, testh, col="green")



new1 <- matrix(x1 <- c(.1,.23),nrow=1)
h <-  runif(2)
new2 <- new1+h
iox_mat(new1, new2, cx, philist)
iox_mat_svd(new1, new2, cx, philist)
iox_mat_svd(new2, new1, cx, philist) %>% t()

iox_mat_svd(new2, new1, cx, philist) %>% t()
[,1]      [,2]
[1,] 0.1145687 0.5488148
[2,] 0.6278148 0.4856909

# C(Ts, Tu) should be dimension 5*q x 2*q

Csu <- rbind(
  cbind(iox(Ts, Tu, 1, 1, cx, philist), iox(Ts, Tu, 1, 2, cx, philist)), 
  cbind(iox(Ts, Tu, 2, 1, cx, philist), iox(Ts, Tu, 2, 2, cx, philist)) 
  )


# PP compare
lags <- seq(1e-4, 0.1, length.out=100)

# spatial
xx <- seq(0, 1, length.out=15)
coords <- expand.grid(xx, xx)
cx <- as.matrix(coords)

plot(lags %>% sapply(\(h) {
  x <- matrix(c(.5001), ncol=2)
  lagh <- matrix(c(h, h), ncol=2)
  cx <- rbind(cx, x, x+lagh)
  
  return(iox(x, x + lagh, 1, 2, cx, c(1,10), 1.9))
}), type='l')


exp(-prod(philist) * as.matrix(dist(rbind(x,x-lagh))))

Llist[[1]][4,,drop=F] %*% t(Llist[[2]][4,,drop=F])

exp(-philist[1] * as.matrix( dist( rbind(x, x + lagh) ) )^1.5) *
  exp(-philist[2] * as.matrix( dist( rbind(x, x + lagh) ) )^1.5)

iox(cx[2,,drop=F], cx[5,,drop=F], 1, , cx, philist)
exp(-philist[1] * as.matrix( dist( cx[c(2,5),] ) )^1.5)









# cross-covar

r <- 3
s <- 5
t(Lilist[[1]][,r,drop=F]) %*% Lilist[[2]][,s,drop=F] 

rhoi <- Clist[[1]][r,,drop=F]
rhoj <- Clist[[2]][,s,drop=F]
rhoi %*% t(Lilist[[1]]) %*% Lilist[[2]] %*% rhoj

iox(cx[r,,drop=F], cx[s,,drop=F], 1, 2, cx, philist, 2)
Llist[[1]][r,,drop=F] %*% t(Llist[[2]][s,,drop=F])


L <- Llist[[1]]
Li <- Lilist[[1]]

c(-Li[r,1:(r-1),drop=F] %*% solve(Li[1:(r-1),1:(r-1)]), 1) / Li[r,r]

Clist[[1]][r,] %*% solve(Clist[[1]]) %*% L %*% t(Li)  %*% t(Li)
L[r,]

crossprod(Lilist[[1]], Lilist[[2]])[r,s]



