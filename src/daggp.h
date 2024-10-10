#ifndef DAGGP 
#define DAGGP

#include <RcppArmadillo.h>

#include "nnsearch.h"
#include "covariance.h"

using namespace std;

class DagGP {
public:
  int nr;
  arma::mat coords;
  arma::vec theta;
  
  double precision_logdeterminant;
  double logdens(const arma::vec& x);
  void update_theta(const arma::vec& newtheta, bool update_H=true);
  arma::sp_mat H;
  void initialize_H();
  
  arma::field<arma::uvec> dag;
  
  // compute and store markov blanket
  arma::field<arma::uvec> mblanket;
  
  // storing just the nonzero elements of rows of H
  arma::field<arma::uvec> ax;
  arma::field<arma::vec> hrows; 
  arma::vec sqrtR;
  arma::field<arma::vec> h;
  void compute_comps(bool update_H=false);
  arma::mat H_times_A(const arma::mat& A, bool use_spmat=true);
  
  // info about covariance model:
  // 0 = power exponential
  // anything else = matern
  int covar;
  double * bessel_ws;
  
  //double ldens;
  DagGP(){};
  
  DagGP(
    const arma::mat& coords_in, 
    const arma::vec& theta_in,
    const arma::field<arma::uvec>& custom_dag,
    int covariance_model=1,
    int num_threads_in=1);
  
  // utils
  arma::uvec oneuv;
  int n_threads;
  
  arma::mat Corr_export(const arma::mat& cx, const arma::uvec& ix, const arma::uvec& jx, bool same);
};




#endif