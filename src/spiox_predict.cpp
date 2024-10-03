#include "spiox.h"

//[[Rcpp::export]]
Rcpp::List spiox_predict(
                   const arma::mat& X_new,
                   const arma::mat& coords_new,
                   
                   const arma::mat& Y, 
                   const arma::mat& X, 
                   const arma::mat& coords,
                   
                   const arma::field<arma::uvec>& dag,
                   
                   const arma::cube& B,
                   const arma::cube& S,
                   const arma::cube& theta,
                   int num_threads = 1
){
  int q = Y.n_cols;
  int p = X.n_cols;
  int mcmc = B.n_slices;
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  double * bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));

  int ntrain = coords.n_rows;
  int ntest = coords_new.n_rows;
  
  arma::mat cxall = arma::join_vert(coords, coords_new);
  
  arma::cube Y_out_mcmc = arma::zeros(ntest, q, mcmc);
  arma::cube random_stdnormal = arma::randn(mcmc, q, ntest);
  
  // loop over test locations
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)
  #endif
  for(int i=0; i<ntest; i++){
    
    arma::uvec ix = oneuv * i + ntrain;
    arma::uvec px = dag(i);
    
    arma::mat theta_current = arma::zeros(4, q);
    
    arma::field<arma::mat> CC(q);
    arma::field<arma::mat> CPt(q);
    arma::field<arma::mat> PPi(q);
    
    for(int m=0; m<mcmc; m++){
      
      arma::mat xb_new = X_new * B.slice(m);
      arma::mat yxb_old = Y - X * B.slice(m);
      
      arma::mat theta_m = theta.slice(m);
      double theta_diff = abs(arma::accu(theta_m - theta_current));
      
      arma::mat Sigmalchol = arma::trans(S.slice(m));
      
      //arma::mat W_out = arma::zeros(ntest, q);
      arma::mat Y_out = arma::zeros(ntest, q);
      arma::mat rndnorm_m = random_stdnormal.row(m);
    
      arma::vec Dj = arma::zeros(q);
      
      // loop over outcomes
      for(int j=0; j<q; j++){
        // predict spatially
        arma::uvec outjx = oneuv * j;

        if(theta_diff > 1e-15){
          CC(j) = Correlationf(cxall, ix, ix, theta_m.col(j), bessel_ws, 1, true); // 1 for matern 
          CPt(j) = Correlationf(cxall, px, ix, theta_m.col(j), bessel_ws, 1, false);
          PPi(j) = arma::inv_sympd( Correlationf(cxall, px, px, theta_m.col(j), bessel_ws, 1, true) );  
        }
        
        arma::vec ht = PPi(j) * CPt(j);
        double sqrtR = sqrt( abs(arma::conv_to<double>::from(
          CC(j) - CPt(j).t() * ht )) ); // abs for numerical zeros
        
        Y_out(i, j) = xb_new(i, j) + 
          arma::conv_to<double>::from(ht.t() * yxb_old(px, outjx)); // post mean only
        Dj(j) = sqrtR;
      }
      
      Y_out.row(i) += arma::trans( arma::diagmat(Dj) * Sigmalchol * rndnorm_m.col(i) ); // pred uncert
      Y_out_mcmc.subcube(i, 0, m, i, q-1, m) = arma::trans(Y_out.row(i));
      
      theta_current = theta_m;
    }
    
    //Rcpp::Rcout << "saving m" << endl;
    //Y_out_mcmc.slice(m) = Y_out;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Y") = Y_out_mcmc,
    Rcpp::Named("dag") = dag
  );
}

/*

 //[[Rcpp::export]]
 Rcpp::List spiox_predict(
 const arma::mat& X_new,
 const arma::mat& coords_new,
 
 const arma::mat& Y, 
 const arma::mat& X, 
 const arma::mat& coords,
 
 const arma::field<arma::uvec>& dag,
 
 const arma::cube& B,
 const arma::cube& S,
 const arma::cube& theta,
 int num_threads = 1
 ){
 int q = Y.n_cols;
 int p = X.n_cols;
 int mcmc = B.n_slices;
 
 arma::uvec oneuv = arma::ones<arma::uvec>(1);
 
 int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
 double * bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
 
 int ntrain = coords.n_rows;
 int ntest = coords_new.n_rows;
 
 arma::mat cxall = arma::join_vert(coords, coords_new);
 
 arma::uvec layers;
 arma::field<arma::uvec> predict_dag = 
 radgpbuild_testset(coords, coords_new, radgp_rho);
 arma::uvec pred_order = sort_test(coords, coords_new);
 
 arma::cube Y_out_mcmc = arma::zeros(ntest, q, mcmc);
 arma::cube W_out_mcmc = arma::zeros(ntest, q, mcmc);
 arma::cube random_stdnormal = arma::randn(mcmc, q, ntest);
 
 // loop over test locations
 for(int i=0; i<ntest; i++){
 Rcpp::Rcout << 100 * i / (ntest+.0) << "%" << endl;
 
 // loop over mcmc iterations. can do parallel
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
 for(int m=0; m<mcmc; m++){
 
 arma::mat xb_new = X_new * B.slice(m);
 arma::mat yxb_old = Y - X * B.slice(m);
 
 arma::mat theta_sample = theta.slice(m);
 arma::mat Sigmalchol = arma::trans(S.slice(m));
 
 //arma::mat W_out = arma::zeros(ntest, q);
 arma::mat Y_out = arma::zeros(ntest, q);
 arma::mat rndnorm_m = random_stdnormal.row(m);
 
 
 
 int itarget = pred_order(i) + ntrain;
 int idagtarget = itarget - ntrain;
 
 arma::vec Dj = arma::zeros(q);
 
 // loop over outcomes
 for(int j=0; j<q; j++){
 // predict spatially
 arma::vec ytemp = arma::zeros(ntrain+ntest);
 ytemp.subvec(0, ntrain-1) = yxb_old.col(j);
 
 arma::uvec ix = oneuv * (itarget);
 arma::uvec px = predict_dag(idagtarget);
 
 arma::mat CC = Correlationf(cxall, ix, ix, theta_sample.col(j), bessel_ws, 1, true); // 1 for matern 
 arma::mat CPt = Correlationf(cxall, px, ix, theta_sample.col(j), bessel_ws, 1, false);
 arma::mat PPi = arma::inv_sympd( Correlationf(cxall, px, px, theta_sample.col(j), bessel_ws, 1, true) );
 
 arma::vec ht = PPi * CPt;
 double sqrtR = sqrt( abs(arma::conv_to<double>::from(
 CC - CPt.t() * ht )) ); // abs for numerical zeros
 
 Y_out(itarget-ntrain, j) = xb_new(itarget-ntrain, j) + 
 arma::conv_to<double>::from(ht.t() * ytemp(px)); // post mean only
 Dj(j) = sqrtR;
 }
 
 Y_out.row(itarget-ntrain) += arma::trans( arma::diagmat(Dj) * Sigmalchol * rndnorm_m.col(i) ); // pred uncert
 Y_out_mcmc.subcube(itarget-ntrain, 0, m, itarget-ntrain, q-1, m) = arma::trans(Y_out.row(itarget-ntrain));
 }
 
 //Rcpp::Rcout << "saving m" << endl;
 //Y_out_mcmc.slice(m) = Y_out;
 }
 
 return Rcpp::List::create(
 Rcpp::Named("Y") = Y_out_mcmc,
 Rcpp::Named("dag") = predict_dag,
 Rcpp::Named("pred_order") = pred_order
 );
 }
 
 */