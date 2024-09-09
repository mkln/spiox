#include <RcppArmadillo.h>
using namespace std;

inline void c_uchol_upd_r1(int n, double *U, double *z) {
  int i, j;
  double r, c, s, temp;
  double alpha=1;

  for (i = 0; i < n; ++i) {
    r = sqrt(U[i * n + i] * U[i * n + i] + alpha * z[i] * z[i]);
    c = r / U[i * n + i];
    s = z[i] / U[i * n + i];
    U[i * n + i] = r;
    
    for (j = i + 1; j < n; ++j) {
      U[j * n + i] = (U[j * n + i] + alpha * s * z[j]) / c;
      z[j] = c * z[j] - s * U[j * n + i];
    }
  }
}

inline void uchol_update(arma::mat& U, const arma::mat& V){
  // Goal: return upper cholesky of U^T U + VV^T
  // where U is the upper chol of an (n x n) spd matrix
  // where V is a (n x p) matrix
  unsigned int n = U.n_rows;
  for(unsigned int i=0; i<V.n_cols; i++){
    arma::vec z = V.col(i);
    c_uchol_upd_r1(n, U.memptr(), const_cast<double*>(z.memptr()));
  }
}
