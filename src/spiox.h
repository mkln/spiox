#include "omp_import.h"
#include "daggp.h"
#include "ramadapt.h"

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart);

class SpIOX {
public:
  // Y = XB + Z where Z is multivariate q,q IOC-GP
  // S^T*S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1. Lambda, Q are sparse
  
  // -------------- data
  
  // matrix of outcomes dim n, q 
  arma::mat Y;
  arma::field<arma::uvec> avail_by_outcome;
  // matrix of predictors dim n, p
  arma::mat X;
  
  // metadata
  unsigned int n, q, p;
  int num_threads;
  //double spatial_sparsity;
  
  // -------------- model parameters
  arma::mat B;
  arma::mat YXB;
  arma::mat B_Var; // prior variance on B, element by element
  
  arma::mat S, Si, Sigma, Q; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // RadGP for spatial dependence
  std::vector<DagGP> daggps, daggps_alt;
  arma::mat theta; // each column is one alternative value for theta
  
  arma::mat V; 
  
  // -------------- utilities
  int matern;
  void sample_B(); // 
  void sample_Sigma_iwishart();
  void compute_V(); 
  void upd_theta_metrop();
  void upd_theta_metrop_conditional();
  void init_theta_adapt();
  
  // latent model 
  int latent_model; // 0: response, 1: block, 2: row seq, 3: col seq
  arma::mat W;
  void gibbs_w_sequential_singlesite();
  void gibbs_w_sequential_byoutcome();
  void gibbs_w_block();
  void sample_Dvec();
  arma::vec Dvec;
  arma::mat XtX;
  //
  
  bool phi_sampling, sigmasq_sampling, nu_sampling, tausq_sampling;
  
  // adaptive metropolis to update theta atoms
  int theta_mcmc_counter;
  arma::uvec which_theta_elem;
  arma::mat theta_unif_bounds;
  //arma::mat theta_metrop_sd;
  RAMAdapt theta_adapt;
  bool theta_adapt_active;
  // --------
  
  // adaptive metropolis (conditional update) to update theta atoms
  // assume shared covariance functions and unknown parameters across variables
  arma::mat c_theta_unif_bounds;
  std::vector<RAMAdapt> c_theta_adapt;
  // --------
  
  
  // -------------- run 1 gibbs iteration based on current values
  void gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta);
  void vi();
  void map();
  double logdens_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructor
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int dag_opts,
        int latent_model_choice,
        const arma::mat& daggp_theta, 
        const arma::mat& Sigma_start,
        const arma::mat& mvreg_B_start,
        int cov_model_matern,
        int num_threads_in) 
  {
    num_threads = num_threads_in;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    latent_model = latent_model_choice; // latent model? 
    
    // managing misalignment, i.e. not all outcomes observed at all locations
    // indices of non-NAs 
    avail_by_outcome = arma::field<arma::uvec>(q);
    //miss_by_outcome = arma::field<arma::uvec>(q);
    for(int j=0; j<q; j++){
      avail_by_outcome(j) = arma::find_finite(Y.col(j));
     // miss_by_outcome(j) = arma::find_nonfinite(Y.col(j));
    }
    if(Y.has_nonfinite() & latent_model!=2){
      Rcpp::stop("nq block and single outcome samplers not implemented for misaligned data.\n");
    }
    
    if(latent_model>0){
      W = Y;
      W(arma::find_nonfinite(W)).fill(0);
      XtX = X.t() * X;
      Dvec = arma::ones(q)*.01;
    }
    
    B = mvreg_B_start;
    YXB = Y - X * B;
    
    B_Var = 1000 * arma::ones(arma::size(B));
    
    theta = daggp_theta;
    daggps = std::vector<DagGP>(q);
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = (q == 1) | (arma::var(theta.row(0)) != 0);
    sigmasq_sampling = (q == 1) | (arma::var(theta.row(1)) != 0);
    nu_sampling = (q == 1) | (arma::var(theta.row(2)) != 0);
    tausq_sampling = (q == 1) | (arma::var(theta.row(3)) != 0);
    
    matern = cov_model_matern;
    //Rcpp::Rcout << "Covariance choice: " << matern << endl;
    
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag, 
                               dag_opts,
                               matern, 
                               latent_model==3, // with q blocks, make Ci
                               num_threads);
    }
    daggps_alt = daggps;
    
    init_theta_adapt();
    
    S = arma::chol(Sigma_start, "upper");
    Si = arma::inv(arma::trimatu(S));
    Sigma = S.t() * S;
    Q = Si * Si.t();
    
    compute_V();
    
    timings = arma::zeros(10);
  }
};
