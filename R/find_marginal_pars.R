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
  
  cat("Processing ", q, " outcomes... \n")
  for(j in seq_len(q)){
    cat(j, " ")
    y_j  <- Y[, j]
    suppressWarnings(out  <- GpGp::fit_model(y = y_j, locs = coords,
                            X = X, m_seq = m, 
                            covfun_name = covfun_name,
                            silent = TRUE))
    # nugget = sigma * tsq
    # spatial variance = sigma
    # total = sigma (1+tsq)
    # alpha (portion of meas error): tsq/(1+tsq)
    if(covfun_name == "matern_isotropic"){
      sigmasq <- out$covparms[1]
      tsq     <- out$covparms[4]
      alpha   <- tsq / (1+tsq)
      phi     <- 1 / out$covparms[2]
      nu      <- out$covparms[3]
    } else {
      sigmasq <- out$covparms[1]
      tsq     <- out$covparms[3]
      alpha   <- tsq / (1+tsq)
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