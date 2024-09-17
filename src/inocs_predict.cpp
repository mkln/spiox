#include "inocs.h"

//[[Rcpp::export]]
Rcpp::List inocs_predict(const arma::mat& coords_new,
                   const arma::mat& X_new,
                   
                   const arma::mat& Y, 
                   const arma::mat& X, 
                   const arma::mat& Xstar,
                   const arma::mat& coords,
                   
                   double radgp_rho, const arma::mat& theta_options, 
                   
                   const arma::cube& B,
                   const arma::cube& S,
                   const arma::umat& theta_which
){
  int q = Y.n_cols;
  int p = X.n_cols;
  int mcmc = B.n_slices;
  
  Inocs inocs_model(Y, X, coords, radgp_rho, theta_options);
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(12*bessel_ws_inc, sizeof(double));
  
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
  
  for(int m=0; m<mcmc; m++){
    Rcpp::Rcout << "m: " << m << endl;
    arma::uvec spmap = theta_which.col(m)-1; // restore c indexing from 0
    
    arma::mat Si = arma::inv(arma::trimatl(S.slice(m)));
    arma::mat W = (inocs_model.Y - X * B.slice(m))*Si;
    
    arma::mat W_out = arma::zeros(ntest, q);
    arma::mat Y_out = arma::zeros(ntest, q);
    
    // loop over test locations
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<ntest; i++){
      // loop over outcomes
      
      int itarget = pred_order(i) + ntrain;
      int idagtarget = itarget - ntrain;
      
      for(int j=0; j<q; j++){
        // predict spatially
        arma::vec wtemp = arma::zeros(ntrain+ntest);
        wtemp.subvec(0, ntrain-1) = W.col(j);
        
        arma::uvec ix = oneuv * (itarget);
        arma::uvec px = predict_dag(idagtarget);
        
        arma::mat CC = inocs_model.radgp_options.at(spmap(j)).Corr_export(cxall, ix, ix, true);
        arma::mat CPt = inocs_model.radgp_options.at(spmap(j)).Corr_export(cxall, px, ix, false);
        arma::mat PPi = arma::inv_sympd( inocs_model.radgp_options.at(spmap(j)).Corr_export(cxall, px, px, true) );
          
        arma::vec ht = PPi * CPt;
        double sqrtR = sqrt( arma::conv_to<double>::from(
          CC - CPt.t() * ht ) );
        
        wtemp(itarget) = arma::conv_to<double>::from(
          ht.t() * wtemp(px) + random_stdnormal(m, j, i) * sqrtR );
        
        W_out(itarget-ntrain, j) = wtemp(itarget);
      
      }
    }
    
    Y_out = W_out * S.slice(m) + X_new * B.slice(m);
    
    //Rcpp::Rcout << "saving m" << endl;
    W_out_mcmc.slice(m) = W_out;
    Y_out_mcmc.slice(m) = Y_out;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Y_spatial") = W_out_mcmc,
    Rcpp::Named("Y") = Y_out_mcmc,
    Rcpp::Named("dag") = predict_dag,
    Rcpp::Named("pred_order") = pred_order
  );
}
