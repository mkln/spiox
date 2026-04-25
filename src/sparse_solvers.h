// pcg_diag_solve.h
#ifndef PCG_DIAG_SOLVE_H
#define PCG_DIAG_SOLVE_H

#include <RcppArmadillo.h>

// Existing matrix-based version: declaration only, defined in the .cpp
arma::vec pcg_diag_solve(
    const arma::sp_mat& A,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    const double diag_floor = 1e-12,
    int num_threads = 1
);

// Matrix-free version: full definition in header (template)
template <typename MV>
arma::vec pcg_diag_solve_mf(
    MV&& mv,
    const arma::vec& b,
    const arma::vec& Mdiag_in,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    const double diag_floor = 1e-12,
    int num_threads = 1
){
  const arma::uword n = b.n_elem;
  arma::vec x = (x0.n_elem == n) ? x0 : arma::zeros(n);
  arma::vec r(n), z(n), p(n), Ap(n);
  
  mv(x, Ap);
  r = b - Ap;
  
  const double bnorm = std::max(arma::norm(b, 2), 1e-300);
  
  arma::vec Mdiag = Mdiag_in;
  for(arma::uword i = 0; i < n; ++i)
    if(!(Mdiag(i) > diag_floor)) Mdiag(i) = diag_floor;
    
    z = r / Mdiag;
    p = z;
    double rz_old = arma::dot(r, z);
    
    int k = 0;
    for(; k < maxit; ++k){
      mv(p, Ap);
      const double denom = arma::dot(p, Ap);
      if(!(denom > 0.0)) break;
      const double alpha = rz_old / denom;
      
      double rr = 0.0;
      double*       xp = x.memptr();
      double*       rp = r.memptr();
      const double* pp = p.memptr();
      const double* ap = Ap.memptr();
#ifdef _OPENMP
#pragma omp parallel for reduction(+:rr) schedule(static) num_threads(num_threads)
#endif
      for(arma::uword i = 0; i < n; ++i){
        xp[i] += alpha * pp[i];
        rp[i] -= alpha * ap[i];
        rr    += rp[i]*rp[i];
      }
      
      if(std::sqrt(rr) / bnorm <= tol){ ++k; break; }
      
      z = r / Mdiag;
      const double rz_new = arma::dot(r, z);
      const double beta = rz_new / rz_old;
      p = z + beta * p;
      rz_old = rz_new;
    }
    return x;
}

#endif