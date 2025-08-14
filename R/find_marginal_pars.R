find_marginal_pars <- function(Y, X, coords, m_nn = 20,
                               ncores = parallel::detectCores(logical = FALSE)-1) {
  
  fitter <- function(j) {
    y_j  <- Y[, j]
    out  <- GpGp::fit_model(y = y_j, locs = coords,
                            X = X, m_seq = m_nn, silent = TRUE)
    
    sigmasq <- out$covparms[1]
    tausq   <- prod(out$covparms[c(1, 4)])
    phi     <- 1 / out$covparms[2]
    nu      <- out$covparms[3]
    
    c(phi, sigmasq, nu, tausq)
  }
  
  res_list <- parallel::mclapply(X = seq_len(ncol(Y)), FUN = fitter,
    mc.cores  = ncores, mc.preschedule = TRUE)
  
  theta <- do.call(cbind, res_list)
  rownames(theta) <- c("phi", "sigmasq", "nu", "tausq")
  return(theta)
}