#include "spf.h"

void SparsePrecisionFactor::compute_P(){
  P = Iq + Lambda.t() * arma::diagmat(1.0/Delta) * Lambda;
  try {
    Pchol = arma::chol(P, "lower"); 
  } catch (...) {
    Rcpp::Rcout << "error at P" << endl
                << P << endl;
    Rcpp::Rcout << 1.0/Delta << endl;
    Rcpp::stop("stop");
  }
  arma::mat Pcholi = arma::inv(arma::trimatl(Pchol));
  Pi = Pcholi.t() * Pcholi;
}

void SparsePrecisionFactor::fc_sample_uv(){
  compute_P();
  
  //Rcpp::Rcout << "u " << arma::size(Pchol) << endl;
  U = arma::randn(n, q) * Pchol.t();
  //Rcpp::Rcout << "v " << arma::size(Delta) << endl;
  V = (*Y) + U * Pi * Lambda.t() * arma::diagmat(1.0/Delta);
}

/*
void SparsePrecisionFactor::fc_sample_Lambda(){
  arma::mat rn = arma::randn(d, q);
  arma::mat VtV = V.t() * V;
  
  // update Lambda by column (parallel OK)
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<q; i++){
    arma::mat post_prec_ichol;
    try {
      post_prec_ichol = 
        arma::inv(arma::trimatl(
            arma::chol(arma::diagmat(1.0/S_Lambda.col(i)) + VtV, "lower") ));  
    } catch (...) {
      Rcpp::Rcout << "error at lambda" << endl
                  << arma::diagmat(1.0/S_Lambda.col(i)) + VtV << endl;
      Rcpp::stop("stop");
    }
    
    // covariance = crossprod of post_prec_ichol
    Lambda.col(i) = post_prec_ichol.t() * (post_prec_ichol * V.t() * U.col(i) + rn.col(i));
  }
}
*/
void SparsePrecisionFactor::fc_sample_Lambda(){
  arma::mat rq = arma::randn(d, q);
  arma::mat rn = arma::randn(n, q);
  arma::mat VtV = V.t() * V;
  
  // update Lambda by column (parallel OK)
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<q; i++){
    // covariance = crossprod of post_prec_ichol
    arma::vec x = V.t() * U.col(i) + arma::diagmat(1.0/sqrt(S_Lambda.col(i))) * rq.col(i) + V.t() * rn.col(i);
    Lambda.col(i) = arma::solve( arma::diagmat(1.0/S_Lambda.col(i)) + VtV, x, 
               arma::solve_opts::likely_sympd + arma::solve_opts::fast );
  }
}

/*
void SparsePrecisionFactor::fc_sample_Lambda_seq(){
  arma::uvec ixd = arma::regspace<arma::uvec>(0, d-1);
  arma::rowvec vssq = arma::sum(V%V, 0);
  // update Lambda by row 
  for(unsigned int i=0; i<d; i++){
    arma::uvec not_i = arma::find(ixd != i);
    arma::mat Uij = U - V.cols(not_i) * Lambda.rows(not_i);
    
    arma::vec wj = arma::zeros(q);
    for(unsigned int ix=0; ix<n; ix++){
      wj += V(ix, i) * arma::trans(Uij.row(ix));
    }
    
    // covariance = crossprod of post_prec_ichol
    arma::rowvec sigma = 1.0/sqrt( 1.0/S_Lambda.row(i) + vssq(i) );
    Lambda.row(i) = arma::trans(
      arma::diagmat(sigma%sigma) * wj + 
        arma::diagmat(sigma) * arma::randn(q) );
  }
}
*/


void SparsePrecisionFactor::fc_sample_Lambda_seq(){
  arma::uvec ixd = arma::regspace<arma::uvec>(0, d-1);
  arma::rowvec vssq = arma::sum(V%V, 0);
  arma::mat UVL = U - V*Lambda;
  
  // update Lambda by row 
  for(unsigned int i=0; i<d; i++){
    //arma::uvec not_i = arma::find(ixd != i);
    UVL += (V.col(i) * Lambda.row(i)); // remove current value from VL, add to U-VL
    
    //arma::mat Uij = arma::diagmat(V.col(i)) * UVL;
    //for(unsigned int r=0; r<Uij.n_rows; r++){
    //  Uij.row(r) *= V(r,i);
    //}
    
    //arma::vec wj = arma::trans(arma::sum(arma::diagmat(V.col(i)) * UVL, 0));
    arma::vec wj = UVL.t() * V.col(i);//arma::diagmat(V.col(i)) * onen;
    
    // covariance = crossprod of post_prec_ichol
    arma::rowvec sigma = 1.0/sqrt( 1.0/S_Lambda.row(i) + vssq(i) );
    Lambda.row(i) = arma::trans(
      arma::diagmat(sigma%sigma) * wj + 
        arma::diagmat(sigma) * arma::randn(q) );
    
    UVL -= (V.col(i) * Lambda.row(i)); // add new value
  }
}

void SparsePrecisionFactor::fc_sample_Delta(){
  for(unsigned int i=0; i<d; i++){
    int a_n = a_delta + n/2.0;
    double b_n = b_delta + arma::accu(pow(V.col(i), 2.0))/2.0;
    
    Delta(i) = R::rgamma(a_n, 1.0/b_n);
  }
}

void SparsePrecisionFactor::fc_sample_dl(){
  arma::vec vlambda = arma::vectorise(Lambda);
  
  arma::vec vS = dl_update_variances(vlambda, a_dl, 2);
  S_Lambda = arma::mat(vS.memptr(), Lambda.n_rows, Lambda.n_cols);
}

