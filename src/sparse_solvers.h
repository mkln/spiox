
#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include <RcppArmadillo.h>

arma::vec pcg_diag_solve(
    const arma::sp_mat& A,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol = 1e-6,
    const int maxit = 500,
    const double diag_floor = 1e-14
);
  
#endif