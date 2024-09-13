#include <RcppArmadillo.h>
#include "covariance.h"
using namespace std;




//[[Rcpp::export]]
arma::mat ioc_xcor(const arma::mat& x, const arma::mat& y, int i, int j,
                   const arma::mat& S, const arma::vec& philist){
  
  arma::vec thetai = arma::ones(4);
  thetai(0) = philist(i-1); // r indexing?
  thetai(3) = 0;
  arma::mat Ki = Correlationc(S, S, thetai, 0, true);
  arma::mat Li = arma::chol(Ki, "lower");
  arma::mat Li_inv = arma::inv(arma::trimatl(Li));
  arma::mat Ki_inv = Li_inv.t() * Li_inv;
  arma::mat rhoi_xS = Correlationc(x, S, thetai, 0, false);
  
  arma::vec thetaj = arma::ones(4);
  thetaj(0) = philist(j-1); // r indexing 
  thetaj(3) = 0;
  arma::mat Kj = Correlationc(S, S, thetaj, 0, true);
  arma::mat Lj = arma::chol(Kj, "lower");
  arma::mat Lj_inv = arma::inv(arma::trimatl(Lj));
  arma::mat Kj_inv = Lj_inv.t() * Lj_inv;
  arma::mat rhoj_yS = Correlationc(y, S, thetaj, 0, false);
  
  arma::mat Rix = arma::zeros(x.n_rows, y.n_rows);
  if(i==j){
    for(unsigned int r=0; r<x.n_rows; r++){
      for(unsigned int c=0; c<y.n_rows; c++){
        double dist = arma::accu(abs( x.row(r) - y.row(c) ));
        if(dist == 0){
          Rix(r,c) = 1 - arma::conv_to<double>::from(
            rhoi_xS.row(r) * Ki_inv * arma::trans(rhoi_xS.row(r)));  
        }
      }
    } 
  }
  
  return rhoi_xS * Li_inv.t() * Lj_inv * rhoj_yS.t() + Rix;
}