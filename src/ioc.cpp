#include <RcppArmadillo.h>
#include "covariance.h"
#include "omp_import.h"

using namespace std;

const double Pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899;

//[[Rcpp::export]]
arma::mat expcov(const arma::mat& x, const arma::mat& y, double phi){
  arma::mat result = arma::zeros(x.n_rows, y.n_rows);
  for(unsigned int i=0; i<x.n_rows; i++){
    for(unsigned int j=0; j<y.n_rows; j++){
      result(i, j) = exp(-phi * arma::accu(pow(x.row(i)-y.row(j), 2.0)));
    }
  }
  return result;
}

//[[Rcpp::export]]
arma::mat iox(const arma::mat& x, const arma::mat& y, int i, int j,
                   const arma::mat& S, const arma::vec& philist, double cexp=1){
  
  arma::vec thetai = arma::ones(4);
  thetai(0) = philist(i-1); // r indexing?
  thetai(2) = cexp; 
  thetai(3) = 0;
  arma::mat Ki = Correlationc(S, S, thetai, 0, true);
  arma::mat Li = arma::chol(Ki, "lower");
  arma::mat Li_inv = arma::inv(arma::trimatl(Li));
  arma::mat Ki_inv = Li_inv.t() * Li_inv;
  arma::mat rhoi_xS = Correlationc(x, S, thetai, 0, false);
  
  arma::vec thetaj = arma::ones(4);
  thetaj(0) = philist(j-1); // r indexing 
  thetaj(2) = cexp;
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

//[[Rcpp::export]]
arma::mat iox_precomp(const arma::mat& x, const arma::mat& y, int i, int j,
                   const arma::field<arma::mat>& Li_invs, 
                   const arma::mat& S, const arma::vec& philist, double cexp=1){
  
  // same as iox but this avoids computing chol/invs
  arma::vec thetai = arma::ones(4);
  thetai(0) = philist(i-1); // r indexing?
  thetai(2) = cexp; // squared exp
  thetai(3) = 0;
  arma::mat Li_inv = Li_invs(i-1); 
  arma::mat Ki_inv; 
  bool done_Ki_inv = false;
  arma::mat rhoi_xS = Correlationc(x, S, thetai, 0, false);
  
  arma::vec thetaj = arma::ones(4);
  thetaj(0) = philist(j-1); // r indexing 
  thetaj(2) = cexp;
  thetaj(3) = 0;
  arma::mat Lj_inv = Li_invs(j-1); 
  arma::mat rhoj_yS = Correlationc(y, S, thetaj, 0, false);
  
  arma::mat Rix = arma::zeros(x.n_rows, y.n_rows);
  if(i==j){
    for(unsigned int r=0; r<x.n_rows; r++){
      for(unsigned int c=0; c<y.n_rows; c++){
        double dist = arma::accu(abs( x.row(r) - y.row(c) ));
        if(dist == 0){
          if(!done_Ki_inv){
            Ki_inv = Li_inv.t() * Li_inv;
            done_Ki_inv = true;
          }
          Rix(r,c) = 1 - arma::conv_to<double>::from(
            rhoi_xS.row(r) * Ki_inv * arma::trans(rhoi_xS.row(r)));  
        }
      }
    } 
  }
  
  return rhoi_xS * Li_inv.t() * Lj_inv * rhoj_yS.t() + Rix;
}


//[[Rcpp::export]]
arma::mat iox_mat(const arma::rowvec& x, const arma::rowvec& y, 
                       const arma::mat& S, const arma::vec& philist, double cexp=1){
  arma::mat result = arma::zeros(philist.n_elem, philist.n_elem);
  for(unsigned int i=0; i<philist.n_elem; i++){
    for(unsigned int j=0; j<philist.n_elem; j++){
      int ir = i+1;
      int jr = j+1;
      result(i, j) = iox(x, y, ir, jr, S, philist, cexp)(0,0);
    }
  }
  return(result);
}


//[[Rcpp::export]]
arma::vec iox_cross_avg(const arma::vec& hlist, int var_i, int var_j,
                        const arma::mat& test_coords, 
                        const arma::mat& S, const arma::vec& philist, 
                        int num_angles=10, double cexp=1){
  
  arma::field<arma::mat> Li_invs(philist.n_elem);
  for(unsigned int i=0; i<philist.n_elem; i++){
    arma::vec thetai = arma::ones(4);
    thetai(0) = philist(i); // r indexing?
    thetai(2) = cexp; 
    thetai(3) = 0;
    arma::mat Ki = Correlationc(S, S, thetai, 0, true);
    arma::mat Li = arma::chol(Ki, "lower");
    Li_invs(i) = arma::inv(arma::trimatl(Li));
  }
  
  arma::mat xcov = arma::mat(test_coords.n_rows, hlist.n_elem);

  // prepare random unif outside omp loop
  arma::vec bigu = arma::randu(hlist.n_elem * test_coords.n_elem * num_angles);
  arma::cube crandu = arma::cube(bigu.memptr(), hlist.n_elem, test_coords.n_elem, num_angles);
  
//#ifdef _OPENMP
//#pragma omp parallel for 
//#endif
  for(unsigned int hi=0; hi<hlist.n_elem; hi++){
    double h = hlist(hi);
    for(unsigned int i=0; i<test_coords.n_rows; i++){
      arma::rowvec x1 = test_coords.row(i);
      
      arma::vec rand_angles = crandu.tube(hi, i) * 2 * Pi;
      for(unsigned int j=0; j<num_angles; j++){
        double angle = rand_angles(j);
        
        arma::rowvec x2 = x1;
        x2(0) += h * cos(angle);
        x2(1) += h * sin(angle);
        
        arma::mat ioxmat = iox_precomp(x1, x2, var_i, var_j, Li_invs, S, philist, cexp);
        xcov(i, hi) += ioxmat(0,0)/(.0+num_angles);
      }
    }
  }
  
  return(arma::trans(arma::mean(xcov, 0)));
}

