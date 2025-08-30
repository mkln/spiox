scaling_factor_at_zero <- function(S, thetamat, m=20, nmax=1000, matern=TRUE, n_threads=1){
  nr <- nrow(S)
  
  subs <- round(seq(1, nr, length.out=min(c(nmax, nr))))
  S_sub <- S[subs,]
  daginfo <- dag_vecchia_maxmin(S_sub, m)
  
  R <- sfact(daginfo$dag, S_sub[daginfo$maxmin,], thetamat, matern, n_threads)
  C <- R
  C[upper.tri(C)] <- t(R)[upper.tri(R)]
  
  return(
    C
  )
}


Sigma_x_sfact <- function(obj, S, ntail=200, m=20, nmax=1000, matern=TRUE, n_threads=1){
  # Sigma and theta are arrays of mcmc samples
  
  nr <- nrow(S)
  mcmc <- dim(obj$Sigma)[3]
  Sigma <- obj$Sigma[,,tail(seq_len(mcmc), ntail)]
  theta <- obj$theta[,,tail(seq_len(mcmc), ntail)]
  
  subs <- round(seq(1, nr, length.out=min(c(nmax, nr))))
  S_sub <- S[subs,]
  daginfo <- spiox:::dag_vecchia_maxmin(S_sub, m)
  
  R <- spiox:::Sigma_x_sfact_cpp(daginfo$dag, S_sub[daginfo$maxmin,], Sigma, theta, matern, n_threads)
  
  return(
    R
  )
}
