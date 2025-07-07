#include <RcppArmadillo.h>
using namespace std;

//[[Rcpp::export]]
arma::cube S_to_Sigma(const arma::cube& S){
  // S is upper cholesky factor of sigma
  arma::cube Sigma(arma::size(S));
  int mcmc = S.n_slices;
  for(int i=0; i<mcmc; i++){
    Sigma.slice(i) = S.slice(i).t() * S.slice(i);
  }
  return Sigma;
}

//[[Rcpp::export]]
arma::cube S_to_Q(const arma::cube& S){
  // S is upper cholesky factor of sigma
  arma::cube Q(arma::size(S));
  int mcmc = S.n_slices;
  for(int i=0; i<mcmc; i++){
    arma::mat Si = arma::inv(arma::trimatu(S.slice(i)));
    Q.slice(i) = Si * Si.t();
  }
  return Q;
}

//[[Rcpp::export]]
arma::cube Sigma_to_correl(const arma::cube& Sigma){
  int q = Sigma.n_rows;
  int k = Sigma.n_cols;
  int m = Sigma.n_slices;
  
  arma::cube Omega = arma::zeros(arma::size(Sigma));
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<m; i++){
    arma::mat dllt = arma::diagmat(1.0/sqrt( Sigma.slice(i).diag() ));
    Omega.slice(i) = dllt * Sigma.slice(i) * dllt;
  }
  return Omega;
}

//[[Rcpp::export]]
arma::cube Sigma_identify(const arma::cube& Sigma, const arma::cube& theta){
  
  arma::cube Sigma_id = arma::cube(arma::size(Sigma));
  for(unsigned int m = 0; m<Sigma.n_slices; m++){
    
    arma::mat theta_local = theta.slice(m);
    arma::mat Ider = arma::diagmat(sqrt(theta_local.row(1)));
    
    Sigma_id.slice(m) = Ider * Sigma.slice(m) * Ider;
  }
  return Sigma_id;
}
