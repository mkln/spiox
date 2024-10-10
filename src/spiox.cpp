#include "spiox.h"
#include "interrupt.h"

//[[Rcpp::export]]
Rcpp::List spiox_wishart(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    const arma::field<arma::uvec>& custom_dag,
                    
                    arma::mat theta_opts, 
                    
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
  
  bool response = true;
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(upd_theta_opts){
    unsigned int n_opts = theta_opts.n_cols;
    if(n_opts > q){
      theta_opts = theta_opts.head_cols(q);
      Rcpp::warning("Limiting theta options to = number of outcomes.\n"
                    "(More options provided for theta than number of outcomes & updating via Metropolis) ");
    }
  }
  
  int sample_precision = 2 * sample_iwish;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, 
                  response,
                  theta_opts, 
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
    
    iox_model.gibbs(m, sample_precision, sample_mvr, sample_theta_gibbs, upd_theta_opts);

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
Rcpp::List spiox_latent(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          arma::mat theta_opts, 
                          
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
  
  bool response = false;
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(upd_theta_opts){
    unsigned int n_opts = theta_opts.n_cols;
    if(n_opts > q){
      theta_opts = theta_opts.head_cols(q);
      Rcpp::warning("Limiting theta options to = number of outcomes.\n"
                      "(More options provided for theta than number of outcomes & updating via Metropolis) ");
    }
  }
  
  int sample_precision = 2 * sample_iwish;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, 
                  response,
                  theta_opts, 
                  Sigma_start,
                  mvreg_B_start,
                  num_threads);
  
  // storage
  arma::cube B = arma::zeros(iox_model.p, q, mcmc);
  arma::cube W = arma::zeros(n, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::umat theta_which = arma::zeros<arma::umat>(q, mcmc);
  arma::cube theta_opts_save = arma::zeros(theta_opts.n_rows, theta_opts.n_cols, mcmc);
  
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs(m, sample_precision, sample_mvr, sample_theta_gibbs, upd_theta_opts);
    
    B.slice(m) = iox_model.B;
    W.slice(m) = iox_model.W;
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
  
  arma::field<arma::sp_mat> Hlist(q);
  for(int i=0; i<q; i++){
    Hlist(i) = iox_model.daggp_options[i].H;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("B") = B,
    Rcpp::Named("Hlist") = Hlist,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("W") = W,
    Rcpp::Named("theta_which") = theta_which,
    Rcpp::Named("theta_opts") = theta_opts_save,
    Rcpp::Named("timings") = iox_model.timings
  );
  
}
