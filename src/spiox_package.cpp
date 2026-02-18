#include "spiox.h"
#include "interrupt.h"

// [[Rcpp::export]]
Rcpp::List spiox_response(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    const arma::field<arma::uvec>& custom_dag,
                
                    const arma::mat& Beta_start,
                    const arma::mat& Sigma_start,
                    const arma::mat& Theta_start, 
                    
                    int mcmc = 1000,
                    int print_every = 100,
                    int matern = 1,
                    int dag_opts = 0,
                    bool sample_Beta = true,
                    bool sample_Sigma = true,
                    const arma::uvec& update_Theta = arma::ones<arma::uvec>(4),
                    int num_threads = 1){
  
  Rcpp::Rcout << "GP-IOX response model." << endl;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  int latent_model = 0;
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  int sample_precision = 2 * sample_Sigma;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  // tausq not needed in this model
  arma::vec tausq_not_needed = arma::zeros(q);
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  Beta_start,
                   Sigma_start,
                   Theta_start, 
                   update_Theta,
                   tausq_not_needed,
                   matern,
                   num_threads);
  
  int nmiss = 0;
  if(iox_model.Y_needs_filling){
    nmiss = iox_model.Y_na_indices.n_elem;
  }
  
  // storage
  arma::cube Beta = arma::zeros(iox_model.p, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  
  // only store samples of imputed Y if missing
  arma::mat Y_missing_fill = arma::zeros(nmiss, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  bool theta_needs_updating = arma::any(update_Theta == 1);
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs(m, sample_precision, sample_Beta, theta_needs_updating);

    Beta.slice(m) = iox_model.B;
    Sigma.slice(m) = iox_model.Sigma;
    theta.slice(m) = iox_model.theta;
    if(iox_model.Y_needs_filling){
      Y_missing_fill.col(m) = iox_model.Y(iox_model.Y_na_indices);
    }
    
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
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Theta") = theta,
    Rcpp::Named("Y_missing_samples") = Y_missing_fill,
    Rcpp::Named("Y_missing_indices") = iox_model.Y_na_indices+1,
    Rcpp::Named("timings") = iox_model.timings,
    Rcpp::Named("dag_cache") = iox_model.daggps[0].dag_cache
  );
  
}


// [[Rcpp::export]]
Rcpp::List spiox_latent(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          const arma::mat& Beta_start,
                          const arma::mat& Sigma_start,
                          const arma::mat& Theta_start, 
                          const arma::vec& Ddiag_start,
                          
                          int mcmc=1000,
                          int print_every=100,
                          int matern = 1,
                          int dag_opts = 0,
                          bool sample_Beta=true,
                          bool sample_Sigma=true,
                          bool sample_Ddiag=true,
                          const arma::uvec& update_Theta = arma::ones<arma::uvec>(4),
                          int num_threads = 1, 
                          int sampling=2){
  
  
  if(sampling==0){
    Rcpp::stop("Run the GP-IOX response model via spiox_response()");
  }
  
  Rcpp::Rcout << "GP-IOX latent model, ";
  if(sampling==1){
    Rcpp::Rcout << "nq block sampler (full block sampler)" << endl;
  }
  if(sampling==2){
    Rcpp::Rcout << "n sequential, q block sampler (single-site sampler)" << endl;
  }
  if(sampling==3){
    Rcpp::Rcout << "n block, q sequential sampler (single-outcome sampler)" << endl;
  }
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  int sample_precision = 2 * sample_Sigma;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  sampling,
                  Beta_start,
                  Sigma_start,
                  Theta_start, 
                  update_Theta,
                  Ddiag_start,
                  matern,
                  num_threads);
  
  // storage
  arma::cube Beta = arma::zeros(iox_model.p, q, mcmc);
  arma::cube Sigma = arma::zeros(q, q, mcmc);
  arma::cube theta = arma::zeros(4, q, mcmc);
  arma::mat Ddiag = arma::zeros(q, mcmc);
  arma::cube W = arma::zeros(n, q, mcmc);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting MCMC" << endl;
  }
  
  bool theta_needs_updating = arma::any(update_Theta == 1);
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs(m, sample_precision, sample_Beta, theta_needs_updating, sample_Ddiag);
    
    Beta.slice(m) = iox_model.B;
    Sigma.slice(m) = iox_model.Sigma;
    theta.slice(m) = iox_model.theta;
    Ddiag.col(m) = iox_model.Dvec;
    W.slice(m) = iox_model.W;

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
  
  //arma::sp_mat Ci = iox_model.daggps[0].H.t() * iox_model.daggps[0].H;
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Theta") = theta,
    Rcpp::Named("W") = W,
    Rcpp::Named("Ddiag") = Ddiag,
    Rcpp::Named("timings") = iox_model.timings,
    Rcpp::Named("markov_blanket") = iox_model.daggps[0].mblanket
  );
}


// [[Rcpp::export]]
Rcpp::List spiox_response_vi(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          int dag_opts,
                          const arma::mat& Theta, 
                          
                          const arma::mat& Sigma_start,
                          const arma::mat& Beta_start,
                          
                          int print_every = 0,
                          int matern = 1,
                          int num_threads = 1){
  
  double tol = 1e-5;
  int min_iter = 1;
  int max_iter = 500;
  
  bool do_vi = true;

  Rcpp::Rcout << "GP-IOX response model." << endl;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  int latent_model = 0;
  
  unsigned int p = X.n_cols;
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  // tausq not needed in this model
  arma::vec tausq_not_needed = arma::zeros(q);
  arma::uvec not_updating_theta = arma::zeros<arma::uvec>(4);
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  
                  Beta_start,
                  Sigma_start,
                  Theta, 
                  not_updating_theta,
                  tausq_not_needed,
                  matern,
                  num_threads, do_vi);
  
  // storage
  arma::mat Beta = arma::zeros(p, q);
  arma::mat Sigma = arma::zeros(q, q);
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting VI" << endl;
  }
  
  bool stop=false;
  int i=0;
  while(!stop){
    i ++;
    
    arma::mat Beta_pre = Beta;
    arma::mat Sigma_pre = Sigma;
    
    iox_model.response_vi();
    
    Beta = iox_model.B;
    Sigma = iox_model.Sigma;
    
    double rel_mu_change = arma::norm(Beta - Beta_pre, 2) / (arma::norm(Beta_pre, 2) + 1e-12);
    double rel_sigma_change = arma::norm(Sigma - Sigma_pre, "fro") / (arma::norm(Sigma_pre, "fro") + 1e-12);
    
    if (rel_mu_change < tol && rel_sigma_change < tol && i > min_iter) {
      stop=true;
    }
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(i % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  i << endl;
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  arma::vec vecBeta_UQ = iox_model.Beta_UQ.diag();
  arma::mat Beta_UQ = arma::mat(vecBeta_UQ.memptr(), p, q);
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Beta_UQ") = Beta_UQ,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Sigma_UQ") = iox_model.Sigma_UQ
  );
  
}



// [[Rcpp::export]]
Rcpp::List spiox_latent_vi(const arma::mat& Y, 
                           const arma::mat& X, 
                           const arma::mat& coords,
                           
                           const arma::field<arma::uvec>& custom_dag,
                           int dag_opts,
                           const arma::mat& Theta, 
                           
                           const arma::mat& Sigma_start,
                           const arma::mat& Beta_start,
                           const arma::vec& Ddiag_start,
                           
                           int matern = 1,
                           int num_threads = 1,
                           int print_every = 0,
                           double tol = 1e-2,
                           int max_iter = 500){
  
  // do min_iter iterations at least
  int min_iter = 100; 
  // then check maximum relative change. if it's <tol for this time then stop
  int wait_time_before_stop = 2;
  
  // for artifacts in other subfunctions.
  int latent_model = 2; 
  bool do_vi = true;
  
  // print method
  Rcpp::Rcout << "GP-IOX latent VI model,\n "
              << "n sequential, q block updates (single-site updates)\n"
              << endl;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#else
  if(num_threads > 1){
    Rcpp::warning("num_threads > 1, but source not compiled with OpenMP support.");
    num_threads = 1;
  }
#endif
  
  unsigned int p = X.n_cols;
  unsigned int q = Y.n_cols;
  unsigned int n = Y.n_rows;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  arma::uvec not_updating_theta = arma::zeros<arma::uvec>(4); //
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  
                  Beta_start,
                  Sigma_start,
                  Theta, 
                  not_updating_theta,
                  Ddiag_start, // Need tau_sq for latent VI 
                  matern,
                  num_threads, do_vi);
  
  // storage
  arma::mat Beta = arma::zeros(p, q);
  arma::mat Sigma = arma::zeros(q, q);
  arma::mat W = arma::zeros(n, q);
  arma::vec Ddiag = arma::zeros(q);
  
  // for trace plots
  arma::vec rel_B_store(max_iter, arma::fill::zeros);
  arma::vec rel_Sigma_store(max_iter, arma::fill::zeros);
  arma::vec rel_W_store(max_iter, arma::fill::zeros);
  arma::vec rel_D_store(max_iter, arma::fill::zeros);
  
  
  if(print_every > 0){
    Rcpp::Rcout << "Starting VI" << endl;
  }
  
  bool stop=false; // stopping flag
  int about_to_exit = 0; // counter for how long we've been "good"
  
  int i=0;
  while(!stop){
    i ++;
    
    arma::mat Beta_pre = Beta;
    arma::mat Sigma_pre = Sigma;
    arma::mat W_pre = iox_model.E_W;
    arma::vec D_pre = iox_model.Dvec;
    
    iox_model.latent_vi();
    
    Beta = iox_model.E_B;
    Sigma = iox_model.Sigma;
    W = iox_model.E_W;
    Ddiag = iox_model.Dvec;
    
    // monitoring convergence of the parameters
    arma::vec rel_change = arma::zeros(4);
    rel_change(0) = arma::norm(Beta - Beta_pre, 2) / (arma::norm(Beta_pre, 2)  + 1e-12);
    rel_change(1) = arma::norm(Sigma - Sigma_pre, "fro") / (arma::norm(Sigma_pre, "fro") + 1e-12);
    rel_change(2) = arma::norm(W - W_pre, "fro") / (arma::norm(W_pre, "fro") + 1e-12);
    rel_change(3) = arma::norm(Ddiag - D_pre, 2) / (arma::norm(D_pre, 2) + 1e-12);
    
    double max_rel_change = rel_change.max();
    
    bool print_condition = (print_every>0);
    if(print_condition){
      print_condition = print_condition & (!(i % print_every));
    };
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  i << endl;
    }
    
    // storing for trace plots
    rel_B_store(i - 1) = rel_change(0);
    rel_Sigma_store(i - 1) = rel_change(1);
    rel_W_store(i - 1) = rel_change(2);
    rel_D_store(i - 1) = rel_change(3);
    
    if(i > min_iter){
      if(max_rel_change < tol){
        // we're doing well, prepare to exit
        about_to_exit += 1;
      } else {
        // reset
        about_to_exit = 0;
      }
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  
    // stopping?
    stop = about_to_exit > wait_time_before_stop;
    if (i >= max_iter) stop = true;
  }
  
  arma::vec vecBeta_UQ = iox_model.Beta_UQ.diag() / (i-1.0);
  arma::mat Beta_UQ = arma::mat(vecBeta_UQ.memptr(), p, q);
  
  // cut off the unused(NA) elements
  rel_B_store = rel_B_store.head(i);
  rel_Sigma_store = rel_Sigma_store.head(i);
  rel_W_store = rel_W_store.head(i);
  rel_D_store = rel_D_store.head(i);
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Beta_UQ") = Beta_UQ,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("Sigma_UQ") = iox_model.Sigma_UQ,
    Rcpp::Named("W") = W,
    Rcpp::Named("Ddiag") = Ddiag,
    Rcpp::Named("Ddiag_UQ") = iox_model.Dvec_UQ,
    Rcpp::Named("rel_change") = Rcpp::List::create(
      Rcpp::Named("rel_Beta")  = rel_B_store,
      Rcpp::Named("rel_Sigma") = rel_Sigma_store,
      Rcpp::Named("rel_W")     = rel_W_store,
      Rcpp::Named("rel_Ddiag") = rel_D_store
    ),
    Rcpp::Named("n_iter") = i,
    Rcpp::Named("markov_blanket") = iox_model.daggps[0].mblanket
  );
  
}

