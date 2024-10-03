#include "spiox.h"
#include "interrupt.h"

//[[Rcpp::export]]
Rcpp::List spiox_wishart(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    const arma::field<arma::uvec>& custom_dag,
                    
                    const arma::mat& theta_opts, 
                    
                    const arma::mat& Sigma_start,
                    const arma::mat& mvreg_B_start,
                           
                    int mcmc=1000,
                    int print_every=100,
                    bool sample_iwish=true,
                    bool sample_mvr=true,
                    bool sample_theta_gibbs=true,
                    bool upd_theta_opts=true,
                    int num_threads = 1){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  
  int q = Y.n_cols;
  int n = Y.n_rows;
  
  int sample_precision = 2 * sample_iwish;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, theta_opts, 
                     Sigma_start,
                     mvreg_B_start,
                     num_threads);
  
  // storage
  arma::cube B = arma::zeros(iox_model.p, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::umat theta_which = arma::zeros<arma::umat>(q, mcmc);
  arma::cube theta_opts_save = arma::zeros(theta_opts.n_rows, theta_opts.n_cols, mcmc);
  
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs_response(m, sample_precision, sample_mvr, sample_theta_gibbs, upd_theta_opts);

    B.slice(m) = iox_model.B;
    Sigma.slice(m) = iox_model.S.t() * iox_model.S;
    
    theta_opts_save.slice(m) = iox_model.theta_options;
    
    arma::mat theta_choice = arma::zeros(4, q);
    for(unsigned int j=0; j<q; j++){
      theta_choice.col(j) = iox_model.theta_options.col(iox_model.spmap(j)); 
        
    }
    
    theta.slice(m) = theta_choice;
    theta_which.col(m) = iox_model.spmap;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("theta_which") = theta_which,
    Rcpp::Named("theta_opts") = theta_opts_save,
    Rcpp::Named("timings") = iox_model.timings
  );
  
}

//[[Rcpp::export]]
double spiox_logdens(const arma::mat& Y, 
                     const arma::mat& X, 
                     const arma::mat& coords,
                     
                     const arma::field<arma::uvec>& custom_dag,
                     
                     const arma::mat& theta, 
                     const arma::mat& Sigma,
                     const arma::mat& mvreg_B){
  
  int q = Y.n_cols;
  int n = Y.n_rows;
  
  Rcpp::Rcout << "Preparing..." << endl;
  
  SpIOX iox_model(Y, X, coords, custom_dag, theta, 
                     Sigma,
                     mvreg_B,
                     1);
  
  double ldens = iox_model.logdens_eval();
  return ldens;
}


Rcpp::List spiox_spf(const arma::mat& Y, 
                         const arma::mat& X, 
                         const arma::mat& coords,
                         
                         double radgp_rho, const arma::mat& theta_opts, 
                         
                         int spf_k, double spf_a_delta, double spf_b_delta, double spf_a_dl,
                         
                         const arma::mat& spf_Lambda_start, 
                         const arma::vec& spf_Delta_start,
                         const arma::mat& mvreg_B_start,
                         
                         int mcmc=1000,
                         int print_every=100,
                         bool sample_spf=true,
                         bool sample_mvr=true,
                         bool sample_gp=true,
                         bool upd_opts=true,
                         int num_threads = 1){
  
  int q = Y.n_cols;
  int n = Y.n_rows;
  /*
   
   arma::mat spf_Lambda_start = arma::zeros(q, spf_k);
   arma::vec spf_Delta_start = arma::ones(q);
   arma::mat mvreg_B_start = arma::zeros(X.n_cols, q);
   
   */
  int sample_precision = 1 * sample_spf;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, radgp_rho, theta_opts, 
                    spf_k, spf_a_delta, spf_b_delta, spf_a_dl,
                    spf_Lambda_start, spf_Delta_start,
                    mvreg_B_start,
                    num_threads);
  
  // storage
  arma::cube B = arma::zeros(iox_model.p, q, mcmc);
  arma::mat theta = arma::zeros(q, mcmc);
  arma::umat theta_which = arma::zeros<arma::umat>(q, mcmc);
  arma::cube theta_opts_save = arma::zeros(theta_opts.n_rows, theta_opts.n_cols, mcmc);
  
  arma::cube Lambda = arma::zeros(q, iox_model.spf.q, mcmc); //spf.q = k
  arma::mat Delta = arma::zeros(q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs_response(m, sample_spf, sample_mvr, sample_gp, upd_opts);
    
    B.slice(m) = iox_model.B;
    Lambda.slice(m) = iox_model.spf.Lambda;
    Delta.col(m) = iox_model.spf.Delta;
  
    theta_opts_save.slice(m) = iox_model.theta_options;
    
    arma::mat theta_choice = arma::zeros(1, q);
    for(unsigned int j=0; j<q; j++){
      theta_choice(0, j) = iox_model.theta_options(0, iox_model.spmap(j));
      //submat(0, iox_model.spmap(j), 1, iox_model.spmap(j));
    }
    
    theta.col(m) = theta_choice.t();
    theta_which.col(m) = iox_model.spmap;
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(m % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  m+1 << " of " << mcmc << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("Lambda") = Lambda,
    Rcpp::Named("Delta") = Delta,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("theta_which") = theta_which,
    Rcpp::Named("theta_opts") = theta_opts_save,
    Rcpp::Named("timings") = iox_model.timings
  );
}
