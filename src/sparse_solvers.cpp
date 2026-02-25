#include "sparse_solvers.h"
#include "daggp.h" 

// Simple Preconditioned Conjugate Gradient (PCG) solver for SPD sparse A
// Diagonal (Jacobi) preconditioner: M = diag(A)
// Interface mirrors a typical gauss_seidel_solve(A, b, x0, tol, maxit)
//
// Notes:
// - Requires A symmetric positive definite (SPD) for CG guarantees.
// - Uses elementwise ops; no explicit matrix factorization.
// - Avoids division by zero in preconditioner with a small floor.

arma::vec pcg_diag_solve(
    const arma::sp_mat& A,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    const double diag_floor
){
  const arma::uword n = A.n_rows;
  arma::vec x = x0;
  if(x.n_elem != n) x.zeros(n);
  
  // r = b - A x
  arma::vec r = b - arma::vec(A * x);
  
  const double bnorm = std::max(arma::norm(b, 2), 1e-300); // avoid 0 div
  double relres = arma::norm(r, 2) / bnorm;
  
  // Diagonal preconditioner: z = M^{-1} r, where M = diag(A)
  arma::vec Mdiag(A.diag());
  // ensure no zero/negative (CG assumes SPD so diag should be > 0, but be defensive)
  for(arma::uword i=0; i<n; i++){
    if(!(Mdiag(i) > diag_floor)) Mdiag(i) = diag_floor;
  }
  
  arma::vec z = r / Mdiag;
  arma::vec p = z;
  
  double rz_old = arma::dot(r, z);
  
  int k = 0;
  for(; k < maxit; k++){
    arma::vec Ap = arma::vec(A * p);
    
    const double denom = arma::dot(p, Ap);
    if(!(denom > 0.0)){
      // Breakdown (A not SPD or numerical issue). Return current iterate.
      break;
    }
    
    const double alpha = rz_old / denom;
    
    x += alpha * p;
    r -= alpha * Ap;
    
    relres = arma::norm(r, 2) / bnorm;
    if(relres <= tol){
      k++; // count this iteration as completed
      break;
    }
    
    z = r / Mdiag;
    const double rz_new = arma::dot(r, z);
    
    const double beta = rz_new / rz_old;
    p = z + beta * p;
    
    rz_old = rz_new;
  }

  return x;
}

