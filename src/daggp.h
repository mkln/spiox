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
  arma::uvec layers;
  arma::vec theta;
  
  int type; // 1=radgp, 2=nn maxmin order
  int M;
  double rho;
  
  double precision_logdeterminant;
  double logdens(const arma::vec& x);
  void update_theta(const arma::vec& newtheta);
  arma::sp_mat H;
  void initialize_H();
  
  arma::field<arma::uvec> dag;
  
  // storing just the nonzero elements of rows of H
  arma::field<arma::uvec> ax;
  arma::field<arma::vec> hrows; 
  void compute_comps();
  arma::mat H_times_A(const arma::mat& A);
  
  // info about covariance model:
  // 0 = power exponential
  // anything else = matern
  int covar;
  double * bessel_ws;
  
  //double ldens;
  DagGP(){};
  
  DagGP(const arma::mat& coords, 
        const arma::vec& theta_in,
        double rho, 
    int covariance_model=1, // matern
    int num_threads_in=1);
  
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