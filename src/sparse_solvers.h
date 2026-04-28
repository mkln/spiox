// pcg_diag_solve.h
#ifndef PCG_DIAG_SOLVE_H
#define PCG_DIAG_SOLVE_H

#include <RcppArmadillo.h>

arma::vec pcg_diag_solve(
    const arma::sp_mat& A,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    const double diag_floor = 1e-12,
    int num_threads = 1
);

// Matrix-free PCG. apply_Minv computes z = M^{-1} r; M must be SPD.
template <typename MV, typename PC>
arma::vec pcg_mf(
    MV&& mv,
    PC&& apply_Minv,
    int& k,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    int num_threads = 1
){
  const arma::uword n = b.n_elem;
  arma::vec x = (x0.n_elem == n) ? x0 : arma::zeros(n);
  arma::vec r(n), z(n), p(n), Ap(n), Mb(n);
  
  mv(x, Ap);
  r = b - Ap;
  
  apply_Minv(b, Mb);
  const double b_prec_safe = std::max(std::sqrt(std::max(arma::dot(b, Mb), 0.0)), 1e-300);
  
  apply_Minv(r, z);
  p = z;
  double rz_old = arma::dot(r, z);
  
  double target = std::sqrt(std::max(rz_old, 0.0)) / b_prec_safe;
  if(target <= tol){ Rcpp::Rcout << 0 << " " << target << "\n"; return x; }
  
  k = 0;
  for(; k < maxit; ++k){
    mv(p, Ap);
    const double denom = arma::dot(p, Ap);
    if(!(denom > 0.0)) break;
    const double alpha = rz_old / denom;
    
    double*       xp = x.memptr();
    double*       rp = r.memptr();
    const double* pp = p.memptr();
    const double* ap = Ap.memptr();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for(arma::uword i = 0; i < n; ++i){
      xp[i] += alpha * pp[i];
      rp[i] -= alpha * ap[i];
    }
    
    apply_Minv(r, z);
    const double rz_new = arma::dot(r, z);
    
    target = std::sqrt(std::max(rz_new, 0.0)) / b_prec_safe;
    if(target <= tol){ ++k; break; }
    
    const double beta = rz_new / rz_old;
    p = z + beta * p;
    rz_old = rz_new;
  }
  
  return x;
}

arma::mat solve_sparse_lower_matrix(const arma::sp_mat& H, const arma::mat& rhs);

arma::mat solve_sparse_upper_matrix(const arma::sp_mat& U, const arma::mat& rhs);
  
#endif