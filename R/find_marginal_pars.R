find_marginal_pars <- function(Y, X, coords, m_nn = 20) {
  
  q <- ncol(Y)
  theta <- matrix(0, nrow=4, ncol=q)
  for(j in seq_len(q)){
    cat(j, "\n")
    y_j  <- Y[, j]
    out  <- GpGp::fit_model(y = y_j, locs = coords,
                            X = X, m_seq = m_nn, silent = TRUE)
    
    sigmasq <- out$covparms[1]
    tausq   <- prod(out$covparms[c(1, 4)])
    phi     <- 1 / out$covparms[2]
    nu      <- out$covparms[3]
    
    theta[,j] <- c(phi, sigmasq, nu, tausq)
  }
  
  rownames(theta) <- c("phi", "sigmasq", "nu", "tausq")
  return(theta)
}