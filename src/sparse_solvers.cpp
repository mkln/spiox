#include "sparse_solvers.h"
#include "daggp.h" 


arma::vec pcg_diag_solve(
    const arma::sp_mat& A,
    const arma::vec& b,
    const arma::vec& x0,
    const double tol,
    const int maxit,
    const double diag_floor,
    int num_threads
){
  const arma::uword n = A.n_rows;
  
  // CSC arrays -- reused as CSR because A is symmetric
  A.sync();
  const double*      vals = A.values;
  const arma::uword* rows = A.row_indices;
  const arma::uword* cols = A.col_ptrs;
  
  // Parallel symmetric SpMV:  y = A x
  auto spmv = [&](const arma::vec& xin, arma::vec& yout){
    const double* xp = xin.memptr();
    double*       yp = yout.memptr();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
    for(arma::uword i = 0; i < n; ++i){
      double yi = 0.0;
      const arma::uword ks = cols[i];
      const arma::uword ke = cols[i+1];
      for(arma::uword k = ks; k < ke; ++k){
        yi += vals[k] * xp[ rows[k] ];
      }
      yp[i] = yi;
    }
  };
  
  arma::vec x = (x0.n_elem == n) ? x0 : arma::zeros(n);
  arma::vec r(n), z(n), p(n), Ap(n);
  
  spmv(x, Ap);          // Ap = A x0
  r = b - Ap;
  
  const double bnorm = std::max(arma::norm(b, 2), 1e-300);
  
  arma::vec Mdiag(A.diag());
  for(arma::uword i=0; i<n; i++) if(!(Mdiag(i) > diag_floor)) Mdiag(i) = diag_floor;
  
  z = r / Mdiag;
  p = z;
  double rz_old = arma::dot(r, z);
  
  int k = 0;
  for(; k < maxit; k++){
    spmv(p, Ap);
    
    const double denom = arma::dot(p, Ap);
    if(!(denom > 0.0)) break;
    
    const double alpha = rz_old / denom;
    
    // Fused update: x += alpha p ; r -= alpha Ap ; and compute ||r|| on the fly
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
      rr    += rp[i] * rp[i];
    }
    
    if(std::sqrt(rr) / bnorm <= tol){ k++; break; }
    
    z = r / Mdiag;
    const double rz_new = arma::dot(r, z);
    const double beta = rz_new / rz_old;
    p = z + beta * p;
    rz_old = rz_new;
  }
  
  // (optional) 
  //Rcpp::Rcout << "[pcg] iters=" << k << "\n";
  return x;
}

arma::mat solve_sparse_lower_matrix(const arma::sp_mat& H, const arma::mat& rhs) {
  if (H.n_rows != H.n_cols) {
    throw std::invalid_argument("Matrix H must be square.");
  }
  if (H.n_rows != rhs.n_rows) {
    throw std::invalid_argument("Row dimensions of H and RHS must match.");
  }
  
  arma::mat Y = rhs; // Copy rhs into our solution matrix
  const arma::uword n = H.n_cols;
  
  // Iterate column by column of H (Highly efficient for CSC format)
  for (arma::uword j = 0; j < n; ++j) {
    arma::sp_mat::const_col_iterator it = H.begin_col(j);
    arma::sp_mat::const_col_iterator it_end = H.end_col(j);
    
    double diag_val = 0.0;
    
    // 1. Locate the diagonal element H(j, j)
    for (arma::sp_mat::const_col_iterator diag_it = it; diag_it != it_end; ++diag_it) {
      if (diag_it.row() == j) {
        diag_val = *diag_it;
        break;
      }
    }
    
    // Check for singularity
    if (std::abs(diag_val) < 1e-14) {
      throw std::runtime_error("Zero encountered on the diagonal. Matrix is singular.");
    }
    
    // 2. Solve for the current row of variables
    // We scale the entire j-th row of the solution matrix
    Y.row(j) /= diag_val;
    
    // 3. Scatter the results to update the remaining rows below row j
    for (; it != it_end; ++it) {
      arma::uword i = it.row();
      if (i > j) { 
        // Subtract the scaled row j from row i
        Y.row(i) -= (*it) * Y.row(j);
      }
    }
  }
  
  return Y;
}

arma::mat solve_sparse_upper_matrix(const arma::sp_mat& U, const arma::mat& rhs) {
  if (U.n_rows != U.n_cols) {
    throw std::invalid_argument("Matrix U must be square.");
  }
  if (U.n_rows != rhs.n_rows) {
    throw std::invalid_argument("Row dimensions of U and RHS must match.");
  }
  
  arma::mat Y = rhs;
  const arma::uword n = U.n_cols;
  
  // Iterate columns of U from last to first (back substitution).
  // CSC layout still gives efficient column access; the only change vs. the
  // lower-triangular case is the direction (n-1 down to 0) and which rows
  // get scattered (rows i < j, i.e. above the diagonal).
  for (arma::uword jj = n; jj-- > 0; ) {
    arma::sp_mat::const_col_iterator it     = U.begin_col(jj);
    arma::sp_mat::const_col_iterator it_end = U.end_col(jj);
    
    double diag_val = 0.0;
    
    // 1. Locate the diagonal element U(jj, jj)
    for (arma::sp_mat::const_col_iterator diag_it = it; diag_it != it_end; ++diag_it) {
      if (diag_it.row() == jj) {
        diag_val = *diag_it;
        break;
      }
    }
    
    if (std::abs(diag_val) < 1e-14) {
      throw std::runtime_error("Zero encountered on the diagonal. Matrix is singular.");
    }
    
    // 2. Solve for the current row of variables
    Y.row(jj) /= diag_val;
    
    // 3. Scatter the results to update the remaining rows above row jj
    for (; it != it_end; ++it) {
      arma::uword i = it.row();
      if (i < jj) {
        Y.row(i) -= (*it) * Y.row(jj);
      }
    }
  }
  
  return Y;
}