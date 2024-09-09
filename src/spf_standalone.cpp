#include "spf.h"
using namespace std;


//[[Rcpp::export]]
Rcpp::List run_spf_model(arma::mat& Y,
                         unsigned int n_factors, 
                         double delta_gamma_shape,
                         double delta_gamma_rate,
                         double dl_dirichlet_a,
                         const arma::mat& Lambda_start,
                         const arma::vec& Delta_start,
                         unsigned int mcmc=1000,
                         int print_every=1000,
                         bool seq_lambda=false){
  
  SparsePrecisionFactor spf(&Y, n_factors, 
                          delta_gamma_shape, delta_gamma_rate, dl_dirichlet_a);
  
  arma::cube Lambda_storage = arma::zeros(spf.d, spf.q, mcmc);
  arma::mat Delta_storage = arma::zeros(spf.d, mcmc);
  arma::cube Lambda_dlvar_storage = arma::zeros(spf.d, spf.q, mcmc);
  
  spf.Lambda_start(Lambda_start);
  spf.Delta_start(Delta_start);
  
  for(unsigned int m=0; m<mcmc; m++){
    //Rcpp::Rcout << "uv " << endl;
    spf.fc_sample_uv();
    //Rcpp::Rcout << "lambda " << endl;
    if(seq_lambda){
      spf.fc_sample_Lambda_seq();
    } else {
      spf.fc_sample_Lambda();
    }
    //Rcpp::Rcout << "delta " << endl;
    spf.fc_sample_Delta();
    //Rcpp::Rcout << "dl " << endl;
    spf.fc_sample_dl();
    
    Lambda_storage.slice(m) = spf.Lambda;
    Delta_storage.col(m) = spf.Delta;
    Lambda_dlvar_storage.slice(m) = spf.S_Lambda;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Lambda") = Lambda_storage,
    Rcpp::Named("Delta") = Delta_storage,
    Rcpp::Named("Lambda_dlvar") = Lambda_dlvar_storage
  );
}