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


autostart <- function(Y, X, coords, method=c("response", "latent"), m=15, nu=0.5){
  
  method <- match.arg(method)
  
  if(nu==0.5){
    covfun_name = "exponential_isotropic"
  } else if(nu==1.5) {
    covfun_name = "matern15_isotropic"
  } else {
    covfun_name = "matern_isotropic"
    cat("nu smoothness value != c(0.5, 1.5), fitting matern_isotropic\n")
  }
  
  p <- ncol(X)
  q <- ncol(Y)
  n <- nrow(Y)
  
  Beta <- matrix(0, nrow=p, ncol=q)
  Theta <- matrix(0, nrow=3, ncol=q)
  Sigma <- diag(q)
  
  if(method == "latent"){
    Ddiag <- rep(0, q)
    W <- matrix(0, nrow=n, ncol=q)
    
    # fill missing (just here)
    Yfill <- Y
    if(any(is.na(Y))){
      for(j in seq_len(q)){
        ix_na <- is.na(Yfill[,j])
        n_na <- sum(ix_na)
        mu_j = mean(Yfill[,j], na.rm=TRUE)
        sd_j = sd(Yfill[,j], na.rm=TRUE)
        Yfill[ix_na, j] <- rnorm(n_na, mean = mu_j, sd = sd_j)
      }  
    }
  } else {
    W <- matrix(0, 1, 1)
  }
  
  
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
    
    Beta[,j] <- out$betahat
    if(method == "latent"){
      W[,j] <- Yfill[,j] - X %*% out$betahat
      Sigma[j,j] <- sigmasq
      Ddiag[j] <- sigmasq * tsq
      Theta[,j] <- c(phi, nu, 0)
    } else {
      Sigma[j,j] <- sigmasq * (1+tsq)
      Theta[,j] <- c(phi, nu, alpha)
    }
  }
  
  cat("\n")

  if(method == "response"){
    out <- list(
      Beta = Beta,
      Sigma = Sigma,
      Theta = Theta
    )
  } else {
    out <- list(
      Beta = Beta,
      W = W,
      Sigma = Sigma,
      Theta = Theta,
      Ddiag = Ddiag
    )
  }

  
  return(out)
  
  
}