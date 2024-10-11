#include <RcppArmadillo.h>
#include "covariance.h"
#include "omp_import.h"
#include "daggp.h"


using namespace std;

const double Pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899;

//[[Rcpp::export]]
arma::mat iox(const arma::mat& x, const arma::mat& y, int i, int j,
                   const arma::mat& S, 
                   const arma::mat& theta, 
                   bool matern = true,
                   bool diag_only=false, bool at_limit=false){

  arma::vec thetai = theta.col(i-1);
  arma::mat Ki = Correlationc(S, S, thetai, matern, true);
  arma::mat Li = arma::chol(Ki, "lower");
  arma::mat Li_inv = arma::inv(arma::trimatl(Li));
  arma::mat Ki_inv = Li_inv.t() * Li_inv;
  arma::mat rhoi_xS = Correlationc(x, S, thetai, matern, false);
  
  arma::vec thetaj = theta.col(j-1);
  arma::mat Kj = Correlationc(S, S, thetaj, matern, true);
  arma::mat Lj = arma::chol(Kj, "lower");
  arma::mat Lj_inv = arma::inv(arma::trimatl(Lj));
  arma::mat Kj_inv = Lj_inv.t() * Lj_inv;
  arma::mat rhoj_yS = Correlationc(y, S, thetaj, matern, false);
  
  arma::mat R = arma::zeros(x.n_rows, y.n_rows);
  
  if(!at_limit){
    arma::mat fixcoord = arma::zeros(1, 2);
    double atZerodisti = Correlationc(fixcoord, fixcoord, thetai, matern, true)(0,0);
    double atZerodistj = Correlationc(fixcoord, fixcoord, thetaj, matern, true)(0,0);
    
    for(unsigned int r=0; r<x.n_rows; r++){
      bool Ri_computed = false; // avoid computing until we actually need 
      double Ri = 0;
      for(unsigned int c=0; c<y.n_rows; c++){
        double dist = arma::accu(abs( x.row(r) - y.row(c) ));
        if(dist == 0){
          if(!Ri_computed){ // compute Ri 
            Ri = atZerodisti - arma::conv_to<double>::from(
              rhoi_xS.row(r) * Ki_inv * arma::trans(rhoi_xS.row(r)) );
            if(Ri < 0){ Ri = 0; } // numerical zero, coord is in S
          }
          double Rj = atZerodistj - arma::conv_to<double>::from(
            rhoj_yS.row(c) * Kj_inv * arma::trans(rhoj_yS.row(c)) );
          if(Rj < 0){ Rj = 0; } // numerical zero, coord is in S
          R(r,c) = sqrt(Ri * Rj);
        }
      }
    } 
  }
  
  
  if(diag_only){
    return arma::sum((rhoi_xS * Li_inv.t())%(rhoj_yS * Lj_inv.t()), 1) + R.diag();
  } else {
    return rhoi_xS * Li_inv.t() * Lj_inv * rhoj_yS.t() + R;
  }
}

int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

//[[Rcpp::export]]
arma::mat sfact(const arma::field<arma::uvec>& dag, const arma::mat& S, 
                const arma::mat& theta, bool matern=true, int n_threads=1){
  int n = S.n_rows;
  int q = theta.n_cols;
  arma::mat I1 = arma::eye(n, n);
  arma::mat result = arma::zeros(q,q);
  
  std::vector<DagGP> daggps(q);
  arma::cube Hinvs(n,n,q);
  for(int i=0; i<q; i++){
    daggps[i] = DagGP(S, theta.col(i), dag, 1, n_threads);
    Hinvs.slice(i) = arma::spsolve(daggps[i].H, I1, "lower");
    for(int j=0; j<=i; j++){
      result(i,j) = arma::accu(Hinvs.slice(i)%Hinvs.slice(j))/(n+.0);
    }
  }
  
  return(result);
}


//[[Rcpp::export]]
arma::mat rvec(const arma::mat& x, int i, 
                      const arma::mat& S, 
                      const arma::mat& theta, 
                      bool matern = true){
  
  arma::vec thetai = theta.col(i-1);
  arma::mat rhoi_xS = Correlationc(x, S, thetai, matern, false);
  
  arma::mat Ki = Correlationc(S, S, thetai, matern, true);
  arma::mat Li = arma::chol(Ki, "lower");
  arma::mat Li_inv = arma::inv(arma::trimatl(Li));
  arma::mat Ki_inv = Li_inv.t() * Li_inv;
  
  arma::vec R = arma::zeros(x.n_rows);
  arma::mat fixcoord = arma::zeros(1, 2);
  double atZerodisti = Correlationc(fixcoord, fixcoord, thetai, matern, true)(0,0);
  
  for(unsigned int r=0; r<x.n_rows; r++){
    double Ri = atZerodisti - arma::conv_to<double>::from(
      rhoi_xS.row(r) * Ki_inv * arma::trans(rhoi_xS.row(r)) );
    if(Ri < 0){ Ri = 0; } // numerical zero, coord is in S
    R(r) = sqrt(Ri);
  } 
  return R;
}


arma::mat iox_precomp(const arma::mat& x, const arma::mat& y, int i, int j,
              const arma::mat& S, 
              const arma::mat& theta, 
              const arma::field<arma::mat>& L_invs,
              bool matern = true,
              bool diag_only=false,
              bool D_only=false){
  
  arma::vec thetai = theta.col(i-1);
  arma::mat rhoi_xS = Correlationc(x, S, thetai, matern, false);
  
  arma::vec thetaj = theta.col(j-1);
  arma::mat rhoj_yS = Correlationc(y, S, thetaj, matern, false);
  
  arma::mat Ki_inv = L_invs(i-1).t() * L_invs(i-1);
  arma::mat Kj_inv = L_invs(j-1).t() * L_invs(j-1);
  
  arma::mat R = arma::zeros(x.n_rows, y.n_rows);
  arma::mat fixcoord = arma::zeros(1, 2);
  double atZerodisti = Correlationc(fixcoord, fixcoord, thetai, matern, true)(0,0);
  double atZerodistj = Correlationc(fixcoord, fixcoord, thetaj, matern, true)(0,0);
  for(unsigned int r=0; r<x.n_rows; r++){
    double Ri = atZerodisti - arma::conv_to<double>::from(
      rhoi_xS.row(r) * Ki_inv * arma::trans(rhoi_xS.row(r)) );
    if(Ri < 0){ Ri = 0; } // numerical zero, coord is in S
    for(unsigned int c=0; c<y.n_rows; c++){
      double dist = arma::accu(abs( x.row(r) - y.row(c) ));
      if(dist == 0){
        double Rj = atZerodistj - arma::conv_to<double>::from(
          rhoj_yS.row(c) * Kj_inv * arma::trans(rhoj_yS.row(c)) );
        if(Rj < 0){ Rj = 0; } // numerical zero, coord is in S
        R(r,c) = sqrt(Ri * Rj);
      }
    }
  } 
  if(D_only){
    return R;
  }
  
  if(diag_only){
    return arma::sum((rhoi_xS * L_invs(i-1).t())%(rhoj_yS * L_invs(j-1).t()), 1) + R.diag();
  } else {
    return rhoi_xS * L_invs(i-1).t() * L_invs(j-1) * rhoj_yS.t() + R;
  }
}

//[[Rcpp::export]]
arma::uvec make_ix(int q, int n){
  return arma::regspace<arma::uvec>(0, n, n*q-1);
}

//[[Rcpp::export]]
arma::mat iox_mat(const arma::mat& x, const arma::mat& y, 
                  const arma::mat& S, const arma::mat& theta, 
                  bool matern = true, bool D_only=false){
  
  int nx = x.n_rows;
  int ny = y.n_rows;
  int q = theta.n_cols;
  
  arma::field<arma::mat> L_invs(q);
  for(unsigned int i=0; i<q; i++){
    arma::vec thetai = theta.col(i);
    arma::mat Ki = Correlationc(S, S, thetai, matern, true);
    arma::mat Li = arma::chol(Ki, "lower");
    L_invs(i) = arma::inv(arma::trimatl(Li));
  }
  
  arma::mat result = arma::zeros(nx * q, ny * q);
  for(unsigned int ni=0; ni<nx; ni++){
    for(unsigned int nj=0; nj<ny; nj++){
      for(unsigned int i=0; i<q; i++){
        for(unsigned int j=0; j<q; j++){
          int ir = i+1;
          int jr = j+1;
          arma::mat ijres = iox_precomp(x.row(ni), y.row(nj), ir, jr, S, theta, L_invs, matern, false, D_only);
          result(i * nx + ni, j * ny + nj) = ijres(0, 0);
        }
      }
    }
  }
  return(result);
}

