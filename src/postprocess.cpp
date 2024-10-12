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
