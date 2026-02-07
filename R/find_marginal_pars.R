find_marginal_pars <- function(Y, X, coords, m_nn = 20, nu=0.5) {
  
  if(nu==0.5){
    covfun_name = "exponential_isotropic"
  } else if(nu==1.5) {
    covfun_name = "matern15_isotropic"
  } else {
    covfun_name = "matern_isotropic"
    cat("nu smoothness value != c(0.5, 1.5), fitting matern_isotropic\n")
  }
  
  q <- ncol(Y)
  theta <- matrix(0, nrow=4, ncol=q)
  for(j in seq_len(q)){
    cat(j, "\n")
    y_j  <- Y[, j]
    out  <- GpGp::fit_model(y = y_j, locs = coords,
                            X = X, m_seq = m_nn, 
                            covfun_name = covfun_name,
                            silent = TRUE)
    
    if(covfun_name == "matern_isotropic"){
      sigmasq <- out$covparms[1]
      tausq   <- prod(out$covparms[c(1, 4)])
      phi     <- 1 / out$covparms[2]
      nu      <- out$covparms[3]
    } else {
      sigmasq <- out$covparms[1]
      tausq   <- prod(out$covparms[c(1, 3)])
      phi     <- 1 / out$covparms[2]
      nu      <- nu
    }
  
    theta[,j] <- c(phi, sigmasq, nu, tausq)
  }
  
  rownames(theta) <- c("phi", "sigmasq", "nu", "tausq")
  return(theta)
}