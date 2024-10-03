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
xx <- seq(0, 1, length.out=14)
coords <- #cbind(runif(100), runif(100))#
  expand.grid(xx, xx)
cx <- as.matrix(coords)
nr <- nrow(cx)

# multivariate
q <- 3
Q <- rWishart(1, q+2, diag(q))[,,1] #
Sigma <- solve(Q) 


theta_mat <- matrix(1, ncol=q, nrow=4)
theta_mat[1,] <- c(5, 20, 30)[1:q]
theta_mat[2,] <- c(1, 1, 1)[1:q]
theta_mat[3,] <- c(0.5, 1.2, 1.9)[1:q]
theta_mat[4,] <- rep(0, q)


Clist <- 1:q %>% lapply(\(j) spiox::Correlationc(cx, cx, theta_mat[,j], 1, 1))
Llist <- Clist %>% lapply(\(C) t(chol(C)))
Lilist <- Llist %>% lapply(\(L) solve(L))





set.seed(1)
new1 <- matrix(runif(2), ncol=2)
new2 <- matrix(runif(2), ncol=2)



Clist1 <- 1:q %>% lapply(\(j) spiox::Correlationc(rbind(cx, new1), rbind(cx, new1), theta_mat[,j], 1, 1))
Llist1 <- Clist1 %>% lapply(\(C) t(chol(C)))
Lilist1 <- Llist1 %>% lapply(\(L) solve(L))

Lsparse <- Matrix::bdiag(Llist %>% lapply(\(L) { L[sample(x=1:prod(dim(L)), size=round(0.9*prod(dim(L))), replace=F)] <- 0; return(L)}))
Cbig <- Matrix::bdiag(Llist1) %*% (Sigma %x% diag(nr+1)) %*% t(Matrix::bdiag(Llist1))
outx <- c(197, 394, 591) 



H <- Cbig[outx, -outx] %*% solve(Cbig[-outx, -outx])
R <- Cbig[outx, outx] - H %*% Cbig[-outx, outx]

Hmake <- 1:q %>% lapply(\(i) Clist1[[i]][outx[1], -outx[1]] %*% solve(Clist1[[i]][-outx[1], -outx[1]]) )

1:q

Clist2 <- 1:q %>% lapply(\(j) spiox::Correlationc(rbind(cx, new2), rbind(cx, new2), theta_mat[,j], 1, 1))
Llist2 <- Clist2 %>% lapply(\(C) t(chol(C)))
Lilist2 <- Llist2 %>% lapply(\(L) solve(L))


Libig <- 1:q %>% lapply(\(j) {
  Li <- matrix(0, nr + 2, nr + 2)
  Li[1:nr, 1:nr] <- Lilist[[j]]
  Li[nr+1, 1:(nr+1)] <- Lilist1[[j]][nr+1,]
  Li[nr+2, c(1:nr, nr+2)] <- Lilist2[[j]][nr+1,]
  return (Li)                     })

Lbig <- Libig %>% lapply(\(Li) solve(Li))
ixnew <- c(nr+1, nr+2, 2*nr + 2 + 1, 2*nr +2  + 2)
(Matrix::bdiag(Lbig) %*% (Sigma %x% diag(nr+2)) %*% t(Matrix::bdiag(Lbig)))[ixnew, ixnew]


Lbig <- 1:q %>% lapply(\(j) {
  L <- matrix(0, nr + 2, nr + 2)
  L[1:nr, 1:nr] <- Llist[[j]]
  L[nr+1, 1:(nr+1)] <- Llist1[[j]][nr+1,]
  L[nr+2, c(1:nr, nr+2)] <- Llist2[[j]][nr+1,]
  return (L)                     })

ixold <- -ixnew

Cbig <- (Matrix::bdiag(Lbig) %*% ((Sigma) %x% diag(nr+2)) %*% 
    t(Matrix::bdiag(Lbig)))

o1 <- 1:196
o23 <- 197:594

Cond_1_given_23 <- Cbig[o1, o1] - Cbig[o1, o23] %*% solve(Cbig[o23, o23]) %*% Cbig[o23, o1]


Cbig2 <- (Sigma %x% matrix(1, nr+2, nr+2)) * 
  (do.call(rbind, Lbig) %*% t(do.call(rbind, Lbig)))

Csave12_r <- (Cbig[ixnew, ixnew] - Cbig[ixnew, ixold] %*% solve(Cbig[ixold, ixold]) %*% Cbig[ixold, ixnew])
Csave12_m <- Cbig[ixnew, ixnew]

Lbig_sub <- Lbig %>% lapply(\(L) L %>% tail(2))
(Sigma %x% matrix(1, 2, 2)) * (do.call(rbind, Lbig_sub) %*% t(do.call(rbind, Lbig_sub)))

C <- function(x,y,theta){
  spiox::Correlationc(x,y,theta,T,F)
}
R <- function(x,theta){
  C(x,x,theta) - C(x,cx,theta) %*% solve(C(cx,cx,theta)) %*% C(cx,x,theta) 
}


newx <- matrix(runif(10*2), ncol=2)
Cmat <- iox_mat(cx[1:2,], cx[1:2,], cx, theta_mat, matern = T, D_only = F)


for(i in 1:q){
  for(j in 1:q){
    cat( iox(rbind(new1, new2), rbind(new1, new2), i, j, cx, theta_mat) , "\n" )   
  }
}



i <- 1; j <- 1
Sigma[i, j] * 
  (Lbig[[i]][nr+1,,drop=F] %*% t(Lbig[[j]][nr+1,,drop=F]))

Sigma[i, j] * 
  iox(rbind(new1, new2), rbind(new1, new2), i, j, cx, theta_mat, matern = T, diag_only = F, limit = F)

Sigma[i,j] * (C(new2, cx, theta_mat[,i]) %*% t(Lilist[[i]]) %*% Lilist[[j]] %*% C(cx, new2, theta_mat[,j]) + 
  sqrt( R(new2, theta_mat[,i])  * R(new2, theta_mat[,j]) ) ) 


Ciox <- function(x, y, i, j){
  iox(x, y, i, j, cx, theta_mat, matern = T, diag_only = F, limit = F)  
}

Riox <- function(x, i, j){
  Ciox(x,x,i,j) - Ciox(x,cx,i,j) %*% solve(Ciox(cx,cx,i,j)) %*% Ciox(cx,x,i,j)
}
Ciox(rbind(new1, new2), rbind(new1, new2), 1, 2)
Sigma[2, 2] * Riox(rbind(new1, new2), 2, 2)
