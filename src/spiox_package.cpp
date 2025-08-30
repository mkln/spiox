#include "spiox.h"
#include "interrupt.h"

//' @title Spatial Response Model using Gaussian Processes with IOX.
//' @description This function performs Bayesian inference for a spatial response model using a 
//' multivariate GP with Inside-Out Cross-Covariance (IOX). 
//'
//' @param Y A numeric matrix (\eqn{n \times q}) of observed multivariate spatial responses, 
//' where \eqn{n} is the number of spatial locations and \eqn{q} is the number of response variables. 
//' @param X A numeric matrix (\eqn{n \times p}) of predictors corresponding to the observed responses.
//' @param coords A numeric matrix (\eqn{n \times d}) of spatial coordinates, where \eqn{d} is the spatial dimension 
//' (e.g., 2 for latitude and longitude).
//' @param custom_dag An object returned from `spiox::dag_vecchia` 
//' @param theta_opts A numeric matrix specifying options for the correlation parameters (\eqn{\theta}) 
//' used during MCMC updates. The way this is input determines how MCMC works. See details below.
//' @param Sigma_start A numeric matrix (\eqn{q \times q}) specifying the starting value for the IOX covariance matrix 
//' \eqn{\Sigma}.
//' @param Beta_start A numeric matrix (\eqn{p \times q}) specifying the starting values for the regression coefficients.
//' @param mcmc An integer specifying the number of MCMC iterations to perform. 
//' @param print_every An integer specifying the frequency of progress updates during MCMC iterations. Default is 100.
//' @param matern An integer flag for enabling Matérn correlation functions for spatial dependence modeling. Default is 1.
//' Other options: 0=power exponential, 2=wave.
//' @param sample_sigma A logical value indicating whether to sample the covariance matrix (\eqn{\Sigma}) 
//' via Gibbs update from an Inverse Wishart prior. Default is TRUE. 
//' @param sample_beta A logical value indicating whether to sample multivariate regression coefficients 
//' via Gibbs update from a Normal prior. Default is TRUE.
//' @param update_theta A logical value indicating whether to update the correlation parameter options (\eqn{\theta}) 
//' adaptively during MCMC iterations. This should be set to TRUE to run "IOX Full" or "IOX Cluster" from the paper, otherwise FALSE.
//' The update is performed jointly for the whole vector if q=3 or less; conditionally in blocks if q>3.
//' @param num_threads An integer specifying the number of threads for parallel computation. Default is 1.
//'
//' @return A list containing:
//' \item{Beta}{Array of dimension (p,q,mcmc) with posterior samples of the regression coefficients (\eqn{\beta}).}
//' \item{Sigma}{Array of dimension (q,q,mcmc) with posterior samples of the covariance matrix (\eqn{\Sigma}).}
//' \item{theta}{Array of dimension (4,q,mcmc) with posterior samples of the correlation parameters (\eqn{\theta}).}
//' \item{theta_which}{Cluster membership for "IOX Grid" and "IOX Cluster".}
//' \item{theta_opts}{Cluster options for "IOX Grid" and "IOX Cluster".}
//' \item{timings}{Breakdown of timings of the various MCMC operations (debugging).}
//'
//' @details The function is designed for scalable inference on spatial multivariate data using GP-IOX. 
//' Use multi-threading (`num_threads > 1`) for faster computation on large datasets.
//' How to set up `theta_opts`. Each column of theta has 4 elements. 1=phi, spatial decay parameter. 2=spatial variance. 3=smoothness or exponent. 4=nugget.
//' MCMC will sample the jth parameter a posteriori (j=1,2,3,4) ONLY if `var(theta_opts[j,])>0`. 
//' In other words, posterior sampling is DISABLED for parameter j if the entire jth row of `theta_opts` is set to the same value (which will be the fixed value of that parameter).
//' for "IOX Full", `theta_opts` should have \eqn{q} columns. For "IOX Cluster", `theta_opts` should have as many columns as the number of clusters (choose it smaller than \eqn{q}).
//' For "IOX Grid", `theta_opts` should have as many columns as the number of elements in the grid. Tested with up to 200 columns.
//' Beware that each column corresponds to a Vecchia-GP with `m` neighbors, so the memory footprint increases linearly with the number of columns of `theta_opts`.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List spiox_response(const arma::mat& Y, 
                    const arma::mat& X, 
                    const arma::mat& coords,
                    
                    const arma::field<arma::uvec>& custom_dag,
                
                    const arma::mat& Beta_start,
                    const arma::mat& Sigma_start,
                    const arma::mat& theta_start, 
                    
                    int mcmc = 1000,
                    int print_every = 100,
                    int matern = 1,
                    int dag_opts = 0,
                    bool sample_Beta = true,
                    bool sample_Sigma = true,
                    const arma::uvec& update_theta = arma::ones<arma::uvec>(4),
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
                   theta_start, 
                   update_theta,
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
  bool theta_needs_updating = arma::any(update_theta == 1);
  
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
    Rcpp::Named("theta") = theta,
    Rcpp::Named("Y_missing_samples") = Y_missing_fill,
    Rcpp::Named("Y_missing_indices") = iox_model.Y_na_indices+1,
    Rcpp::Named("timings") = iox_model.timings,
    Rcpp::Named("dag_cache") = iox_model.daggps[0].dag_cache
  );
  
}

//' @title Spatial Latent Model using Gaussian Processes with IOX as prior for latent effects.
//' @description This function performs Bayesian inference for a spatial latent model using a 
//' multivariate GP with Inside-Out Cross-Covariance (IOX) prior for the latent effects. 
//'
//' @param Y A numeric matrix (\eqn{n \times q}) of observed multivariate spatial responses, 
//' where \eqn{n} is the number of spatial locations and \eqn{q} is the number of response variables. 
//' @param X A numeric matrix (\eqn{n \times p}) of predictors corresponding to the observed responses.
//' @param coords A numeric matrix (\eqn{n \times d}) of spatial coordinates, where \eqn{d} is the spatial dimension 
//' (e.g., 2 for latitude and longitude).
//' @param custom_dag An object returned from `spiox::dag_vecchia` 
//' @param theta_opts A numeric matrix specifying options for the correlation parameters (\eqn{\theta}) 
//' used during MCMC updates. The way this is input determines how MCMC works. See details below.
//' @param Sigma_start A numeric matrix (\eqn{q \times q}) specifying the starting value for the IOX covariance matrix 
//' \eqn{\Sigma}.
//' @param Beta_start A numeric matrix (\eqn{p \times q}) specifying the starting values for the regression coefficients.
//' @param mcmc An integer specifying the number of MCMC iterations to perform. 
//' @param print_every An integer specifying the frequency of progress updates during MCMC iterations. Default is 100.
//' @param matern An integer flag for enabling Matérn correlation functions for spatial dependence modeling. Default is 1.
//' Other options: 0=power exponential, 2=wave.
//' @param sample_sigma A logical value indicating whether to sample the covariance matrix (\eqn{\Sigma}) 
//' via Gibbs update from an Inverse Wishart prior. Default is TRUE. 
//' @param sample_beta A logical value indicating whether to sample multivariate regression coefficients 
//' via Gibbs update from a Normal prior. Default is TRUE.
//' @param sample_theta_gibbs A logical value indicating whether to enable Gibbs sampling for the correlation 
//' parameters (\eqn{\theta}). This should be set to TRUE to run "IOX Grid" or "IOX Cluster" from the paper, otherwise FALSE.
//' @param update_theta A logical value indicating whether to update the correlation parameter options (\eqn{\theta}) 
//' adaptively during MCMC iterations. This should be set to TRUE to run "IOX Full" or "IOX Cluster" from the paper, otherwise FALSE.
//' The update is performed jointly for the whole vector if q=3 or less; conditionally in blocks if q>3.
//' @param num_threads An integer specifying the number of threads for parallel computation. Default is 1.
//' @param sampling An integer specifying how to sample the latent effects. Available options:  
//' sampling=1: block sampler for the entire set of latent effects (AVOID if \eqn{n} or \eqn{q} are large)
//' sampling=2 (default): single-outcome block sampler (\eqn{q} sequential steps)
//' sampling=3: single-site sampler (\eqn{n} sequential steps)
//'
//' @return A list containing:
//' \item{Beta}{Array of dimension (p,q,mcmc) with posterior samples of the regression coefficients (\eqn{\beta}).}
//' \item{Sigma}{Array of dimension (q,q,mcmc) with posterior samples of the covariance matrix (\eqn{\Sigma}).}
//' \item{theta}{Array of dimension (4,q,mcmc) with posterior samples of the correlation parameters (\eqn{\theta}).}
//' \item{theta_which}{Cluster membership for "IOX Grid" and "IOX Cluster".}
//' \item{theta_opts}{Cluster options for "IOX Grid" and "IOX Cluster".}
//' \item{W}{Array of dimension (n,q,mcmc) with posterior samples of the latent effects.}
//' \item{Ddiag}{Matrix of dimension (q, mcmc) with posterior samples of the diagonal of measurement error matrix D.}
//' \item{timings}{Breakdown of timings of the various MCMC operations (debugging).}
//'
//' @details The function is designed for scalable inference on spatial multivariate data using GP-IOX. 
//' Use multi-threading (`num_threads > 1`) for faster computation on large datasets.
//' How to set up `theta_opts`. Each column of theta has 4 elements. 1=phi, spatial decay parameter. 2=spatial variance. 3=smoothness or exponent. 4=nugget.
//' MCMC will sample the jth parameter a posteriori (j=1,2,3,4) ONLY if `var(theta_opts[j,])>0`. 
//' In other words, posterior sampling is DISABLED for parameter j if the entire jth row of `theta_opts` is set to the same value (which will be the fixed value of that parameter).
//' for "IOX Full", `theta_opts` should have \eqn{q} columns. For "IOX Cluster", `theta_opts` should have as many columns as the number of clusters (choose it smaller than \eqn{q}).
//' For "IOX Grid", `theta_opts` should have as many columns as the number of elements in the grid. Tested with up to 200 columns.
//' Beware that each column corresponds to a Vecchia-GP with `m` neighbors, so the memory footprint increases linearly with the number of columns of `theta_opts`.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List spiox_latent(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          
                          const arma::mat& Beta_start,
                          const arma::mat& Sigma_start,
                          const arma::mat& theta_start, 
                          const arma::vec& tausq_start,
                          
                          int mcmc=1000,
                          int print_every=100,
                          int matern = 1,
                          int dag_opts = 0,
                          bool sample_sigma=true,
                          bool sample_beta=true,
                          bool sample_tausq=true,
                          const arma::uvec& update_theta = arma::ones<arma::uvec>(4),
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
  
  int sample_precision = 2 * sample_sigma;
  
  if(print_every > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  sampling,
                  Beta_start,
                  Sigma_start,
                  theta_start, 
                  update_theta,
                  tausq_start,
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
  
  bool theta_needs_updating = arma::any(update_theta == 1);
  
  for(unsigned int m=0; m<mcmc; m++){
    
    iox_model.gibbs(m, sample_precision, sample_beta, theta_needs_updating, sample_tausq);
    
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
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma,
    Rcpp::Named("theta") = theta,
    Rcpp::Named("W") = W,
    Rcpp::Named("Ddiag") = Ddiag,
    Rcpp::Named("timings") = iox_model.timings
  );
  
}


// [[Rcpp::export]]
Rcpp::List spiox_response_vi(const arma::mat& Y, 
                          const arma::mat& X, 
                          const arma::mat& coords,
                          
                          const arma::field<arma::uvec>& custom_dag,
                          int dag_opts,
                          const arma::mat& theta, 
                          
                          const arma::mat& Sigma_start,
                          const arma::mat& Beta_start,
                          
                          int verbose = 0,
                          int matern = 1,
                          int num_threads = 1){
  
  double tol = 1e-5;
  int min_iter = 1;
  int max_iter = 500;
  
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
  
  if(verbose > 0){
    Rcpp::Rcout << "Preparing..." << endl;
  }
  
  // tausq not needed in this model
  arma::vec tausq_not_needed = arma::zeros(q);
  arma::uvec not_updating_theta = arma::zeros<arma::uvec>(q);
  
  SpIOX iox_model(Y, X, coords, custom_dag, dag_opts,
                  latent_model,
                  
                  Beta_start,
                  Sigma_start,
                  theta, 
                  not_updating_theta,
                  tausq_not_needed,
                  matern,
                  num_threads);
  
  // storage
  arma::mat Beta = arma::zeros(iox_model.p, q);
  arma::mat Sigma = arma::zeros(q, q);
  
  bool stop=false;
  int i=0;
  while(!stop){
    i ++;
    
    arma::mat Beta_pre = Beta;
    arma::mat Sigma_pre = Sigma;
    
    iox_model.vi();
    
    Beta = iox_model.B;
    Sigma = iox_model.Sigma;
    
    double rel_mu_change = arma::norm(Beta - Beta_pre, 2) / arma::norm(Beta_pre, 2);
    double rel_sigma_change = arma::norm(Sigma - Sigma_pre, "fro") / arma::norm(Sigma_pre, "fro");
    
    if (rel_mu_change < tol && rel_sigma_change < tol && i > min_iter) {
      stop=true;
    }
    
    bool print_condition = (verbose>0);
    if(print_condition){
      Rcpp::Rcout << "Iteration: " <<  i << endl;
    };
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Beta") = Beta,
    Rcpp::Named("Sigma") = Sigma
  );
  
}

