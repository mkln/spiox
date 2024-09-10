#include "inocs.h"

//[[Rcpp::export]]
Rcpp::List inocs(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    double radgp_rho, const arma::vec& theta_start, 
                    
                    int spf_k, double spf_a_delta, double spf_b_delta, double spf_a_dl,
                           
                           const arma::mat& spf_Lambda_start, 
                           const arma::vec& spf_Delta_start,
                           const arma::mat& mvreg_B_start,
                           
                    int mcmc=1000,
                    int print_every=100,
                    int sample_precision=1,
                    bool sample_mvr=true,
                    bool sample_gp=true){
  
  int q = Y.n_cols;
  int n = Y.n_rows;
  /*
   
   arma::mat spf_Lambda_start = arma::zeros(q, spf_k);
   arma::vec spf_Delta_start = arma::ones(q);
   arma::mat mvreg_B_start = arma::zeros(X.n_cols, q);
   
   */
  
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  Inocs inocs_model(Y, X, coords, radgp_rho, theta_start, 
                     spf_k, spf_a_delta, spf_b_delta, spf_a_dl,
                     spf_Lambda_start, spf_Delta_start,
                     mvreg_B_start);
  
  // storage
  arma::cube B = arma::zeros(inocs_model.p, q, mcmc);
  arma::cube Lambda = arma::zeros(q, inocs_model.spf.q, mcmc); //spf.q = k
  arma::mat Delta = arma::zeros(q, mcmc);
  arma::cube S = arma::zeros(q, q, mcmc);
  arma::cube Si = arma::zeros(q, q, mcmc);
  arma::mat theta = arma::zeros(q, mcmc);
  arma::cube V = arma::zeros(n, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    inocs_model.gibbs_response(sample_precision, sample_mvr, sample_gp);
    
    B.slice(m) = inocs_model.B;
    Lambda.slice(m) = inocs_model.spf.Lambda;
    Delta.col(m) = inocs_model.spf.Delta;
    S.slice(m) = inocs_model.S;
    Si.slice(m) = inocs_model.Si;
    V.slice(m) = inocs_model.spf.V;
    theta.col(m) = inocs_model.theta;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("Lambda") = Lambda,
    Rcpp::Named("Delta") = Delta,
    Rcpp::Named("S") = S,
    Rcpp::Named("Si") = Si,
    Rcpp::Named("V") = V,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("timings") = inocs_model.timings
  );
  
}
