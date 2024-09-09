#ifndef DAGGP 
#define DAGGP

#include <RcppArmadillo.h>

#include "nnsearch.h"
#include "covariance.h"
#include "interrupt.h"

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
  
  arma::sp_mat H, Ci;
  void initialize_Ci();
  
  arma::field<arma::uvec> dag;
  
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
    int covariance_model=0,
    int nthread=0);
  
  // utils
  arma::uvec oneuv;
  int n_threads;
  
  arma::mat Corr_export(const arma::mat& cx, const arma::uvec& ix, const arma::uvec& jx, bool same);
};




#endif