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
    int num_threads = 1,
    const double energy_tol = 0.0,
    const int energy_delay = 10
){
  const arma::uword n = b.n_elem;
  arma::vec x = (x0.n_elem == n) ? x0 : arma::zeros(n);
  arma::vec r(n), z(n), p(n), Ap(n);
  
  mv(x, Ap);
  r = b - Ap;

  // Reference scale for the relative-residual stopping criterion.  Using the
  // L2 norm of b (unpreconditioned) keeps the criterion preconditioner-
  // independent: stopping at ||r||_2 / ||b||_2 < tol guarantees the same
  // *L2 error* in the CG solution regardless of which preconditioner is in
  // use.  The earlier M^{-1}-norm criterion (sqrt(r^T M^{-1} r) / sqrt(b^T M^{-1} b))
  // is preconditioner-dependent: PPCG's M^{-1} = K (prior cov) heavily
  // weights smooth modes, so CG could pass the M-norm tolerance while still
  // having O(1) residual in rough modes — leading to biased Bhattacharya
  // draws.  Jacobi's M^{-1} ≈ I makes its M-norm ≈ L2 norm, which is why
  // Jacobi was already producing correct samples under the old criterion.
  const double b_l2_safe = std::max(std::sqrt(arma::dot(b, b)), 1e-300);

  apply_Minv(r, z);
  p = z;
  double rz_old = arma::dot(r, z);
  const bool   use_energy = (energy_tol > 0.0);
  const double energy_tol2 = energy_tol * energy_tol;
  const int    delay = std::max(1, energy_delay);
  std::vector<double> gamma_win(use_energy ? delay : 0, 0.0);

  k = 0;
  for(; k < maxit; ++k){
    mv(p, Ap);
    const double denom = arma::dot(p, Ap);
    if(!(denom > 0.0)) break;
    const double alpha = rz_old / denom;
    if(use_energy) gamma_win[k % delay] = alpha * rz_old;    
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

    if(use_energy){
      // window sum = estimate of ||e_{k-d+1}||_A^2 >= ||e_{k+1}||_A^2 (x is now x_{k+1})
      if(k + 1 >= delay){
        double win = 0.0;
        for(int j = 0; j < delay; ++j) win += gamma_win[j];
        if(win <= energy_tol2){ ++k; break; }
      }
    } else {
      const double target = std::sqrt(arma::dot(r, r)) / b_l2_safe;
      if(target <= tol){ ++k; break; }
    }
    
    apply_Minv(r, z);
    const double rz_new = arma::dot(r, z);
    
    const double beta = rz_new / rz_old;
    p = z + beta * p;
    rz_old = rz_new;
  }
  
  return x;
}

#endif