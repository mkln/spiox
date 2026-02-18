find_marginal_pars <- function(Y, X, coords, m = 15, nu=0.5, return_full=FALSE) {
  
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
  
  cat("Processing ", q, " outcomes... ")
  for(j in seq_len(q)){
    cat(j, " ")
    y_j  <- Y[, j]
    out  <- GpGp::fit_model(y = y_j, locs = coords,
                            X = X, m_seq = m, 
                            covfun_name = covfun_name,
                            silent = TRUE)
    # nugget = sigma * alpha, like in IOX
    if(covfun_name == "matern_isotropic"){
      sigmasq <- out$covparms[1]
      alpha   <- out$covparms[4]
      phi     <- 1 / out$covparms[2]
      nu      <- out$covparms[3]
    } else {
      sigmasq <- out$covparms[1]
      alpha   <- out$covparms[3]
      phi     <- 1 / out$covparms[2]
      nu      <- nu
    }
  
    theta[,j] <- c(phi, sigmasq, nu, alpha)
  }
  rownames(theta) <- c("phi", "sigmasq", "nu", "alpha")
  cat("\n")
  
  if(!return_full){
    theta <- theta[-2,]
  }
  return(theta)
}