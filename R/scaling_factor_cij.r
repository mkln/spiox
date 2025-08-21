scaling_factor_at_zero <- function(S, thetamat, m=20, nmax=1000, matern=TRUE, n_threads=1){
  nr <- nrow(S)
  
  subs <- round(seq(1, nr, length.out=min(c(nmax, nr))))
  S_sub <- S[subs,]
  daginfo <- dag_vecchia_maxmin(S_sub, m)
  
  return(
    sfact(daginfo$dag, S_sub[daginfo$maxmin,], thetamat, matern, n_threads)
  )
}
