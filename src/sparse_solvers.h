#include <RcppArmadillo.h>


inline arma::vec sparse_forward_substitution(const arma::sp_mat& L, const arma::vec& b)
{
  using arma::uword;
  
  const uword n = L.n_rows;
  
  if (L.n_rows != L.n_cols) {
    throw std::invalid_argument("sparse_forward_substitution: L must be square");
  }
  if (b.n_rows != n) {
    throw std::invalid_argument("sparse_forward_substitution: dimension mismatch");
  }
  
  arma::vec x(n, arma::fill::zeros);
  arma::vec r = b;  // residual we’ll update as we go
  
  for (uword k = 0; k < n; ++k) {
    // --- 1. find diagonal element L(k,k) in column k ---
    double diag = 0.0;
    bool found_diag = false;
    
    for (arma::sp_mat::const_col_iterator it = L.begin_col(k);
         it != L.end_col(k); ++it)
    {
      const uword i = it.row();   // row index
      if (i == k) {
        diag = *it;
        found_diag = true;
        break;  // rows in a column are sorted, so we can stop once i >= k
      } else if (i > k) {
        // since rows are in ascending order, if we've passed k, the diag isn't present
        break;
      }
    }
    
    if (!found_diag || diag == 0.0) {
      throw std::runtime_error("sparse_forward_substitution: zero or missing diagonal");
    }
    
    // --- 2. solve for x_k using the current residual ---
    x(k) = r(k) / diag;
    
    // --- 3. update residual r_i for i > k using column k ---
    for (arma::sp_mat::const_col_iterator it = L.begin_col(k);
         it != L.end_col(k); ++it)
    {
      const uword i = it.row();
      if (i > k) {
        r(i) -= (*it) * x(k);
      }
    }
  }
  
  return x;
}

inline arma::vec sparse_backward_substitution(const arma::sp_mat& U, const arma::vec& b)
{
  const arma::uword n = U.n_rows;
  
  if (U.n_rows != U.n_cols) {
    Rcpp::stop("sparse_backward_substitution: U must be square");
  }
  if (b.n_rows != n) {
    Rcpp::stop("sparse_backward_substitution: dimension mismatch");
  }
  
  arma::vec x(n, arma::fill::zeros);
  
  // Backward sweep: k = n-1, ..., 0
  for (int kk = static_cast<int>(n) - 1; kk >= 0; --kk) {
    const arma::uword k = static_cast<arma::uword>(kk);
    
    double diag = 0.0;
    double rhs  = b(k);  // start with b_k
    
    // Iterate over nonzeros in row k
    for (arma::sp_mat::const_row_iterator it = U.begin_row(k);
         it != U.end_row(k); ++it)
    {
      const arma::uword j = it.col();
      const double val = *it;
      
      if (j == k) {
        // diagonal element A_kk
        diag = val;
      } else if (j > k) {
        // strict upper part: subtract A_kj * x_j
        rhs -= val * x(j);
      } else {
        // j < k: belongs to lower part; ignored in (D+U)
      }
    }
    
    if (diag == 0.0) {
      Rcpp::stop("sparse_backward_substitution: zero diagonal at row %d",
                 kk + 1);
    }
    
    x(k) = rhs / diag;
  }
  
  return x;
}


inline arma::vec gauss_seidel_step(const arma::sp_mat& A, const arma::vec& b, const arma::vec& x_old)
{
  const arma::uword n = A.n_rows;
  
  if (A.n_rows != A.n_cols) {
    Rcpp::stop("gauss_seidel_step: A must be square");
  }
  if (b.n_rows != n || x_old.n_rows != n) {
    Rcpp::stop("gauss_seidel_step: dimension mismatch");
  }
  
  // r = b - U * x_old, where U is the strict upper part of A
  arma::vec r = b;
  
  for (arma::uword j = 0; j < n; ++j) {
    for (arma::sp_mat::const_col_iterator it = A.begin_col(j);
         it != A.end_col(j); ++it)
    {
      const arma::uword i = it.row();
      if (i < j) {  // strict upper part: row < col
        r(i) -= (*it) * x_old(j);
      }
    }
  }
  
  // solve (D+L) x_new = r using the lower-triangular solver
  arma::vec x_new = sparse_forward_substitution(A, r);
  
  return x_new;
}

inline arma::vec gauss_seidel_solve(const arma::sp_mat& A,
                             const arma::vec& b,
                             const arma::vec& xstart,
                             double tol = 1e-6,
                             int max_iter = 1000)
{
  
  const arma::uword n = A.n_rows;
  arma::vec x = xstart;
  
  if (A.n_rows != A.n_cols) {
    Rcpp::stop("gauss_seidel_solve: A must be square");
  }
  if (b.n_rows != n) {
    Rcpp::stop("gauss_seidel_solve: dimension mismatch");
  }
  if (max_iter <= 0) {
    Rcpp::stop("gauss_seidel_solve: max_iter must be positive");
  }
  
  if(max_iter == 1){
    return gauss_seidel_step(A, b, x);
  }
  
  bool converged = false;
  double last_diff = NA_REAL;
  int iters = 0;
  
  for (int k = 0; k < max_iter; ++k) {
    arma::vec x_new = gauss_seidel_step(A, b, x);
    
    // change in solution (2-norm)
    last_diff = arma::norm(x_new - x, 2);
    
    x = std::move(x_new);
    iters = k + 1;
    
    if (last_diff <= tol) {
      converged = true;
      break;
    }
  }
  
  //if(converged){ // as used in spiox, no convergence =approx a single-element sampler
    return x;
  //} else {
  //  Rcpp::stop("Error in GS solver: did not converge within max_iter.\n");
  //}
  
}

inline arma::vec solve_sparse_lower(const arma::sp_mat& H, const arma::vec& b) 
{
  if (H.n_rows != H.n_cols) {
    throw std::invalid_argument("Matrix H must be square.");
  }
  if (H.n_rows != b.n_elem) {
    throw std::invalid_argument("Dimensions of H and b must match.");
  }
  
  arma::vec y = b; // Copy rhs into our solution vector
  const arma::uword n = H.n_cols;
  
  // Iterate column by column (Highly efficient for CSC format)
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
    
    // 2. Solve for the current variable
    y(j) /= diag_val;
    
    // 3. Scatter the results to update the remaining variables below row j
    for (; it != it_end; ++it) {
      arma::uword i = it.row();
      if (i > j) { 
        y(i) -= (*it) * y(j);
      }
    }
  }
  
  return y;
}