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
  bool upd_theta_metrop();
  arma::uvec upd_theta_metrop_conditional(); // returns uvec with changes to thetaj
  void init_theta_adapt();
  
  // data with with misalignment
  arma::field<arma::uvec> avail_by_outcome;
  arma::umat missing_mat;
  arma::uvec rows_with_missing;
  bool Y_needs_filling;
  arma::uvec Y_na_indices;
  void manage_missing_data(){
    // managing misalignment, i.e. not all outcomes observed at all locations
    // indices of non-NAs 
    avail_by_outcome = arma::field<arma::uvec>(q);
    missing_mat = arma::zeros<arma::umat>(n, q);
    arma::uvec row_miss_01 = arma::zeros<arma::uvec>(n);
    Y_na_indices = arma::find_nonfinite(Y);
    for(int j=0; j<q; j++){
      avail_by_outcome(j) = arma::find_finite(Y.col(j));
      for(int i=0; i<n; i++){
        missing_mat(i,j) = !arma::is_finite(Y(i,j));
        if(latent_model==0){
          // response model: rnorm fill missing in Y
          if(missing_mat(i,j)){
            Y(i,j) = arma::randn();
          }
        }
      }
      
    }
    
    for(int i=0; i<n; i++){
      row_miss_01(i) = arma::any(missing_mat.row(i) == 1);
    }
    rows_with_missing = arma::find(row_miss_01 == 1);
    
    if(Y.has_nonfinite() & (latent_model!=2)){
      Rcpp::stop("nq block and single outcome samplers not implemented for misaligned data.\n");
    }
    if(arma::accu(missing_mat)>0){
      Y_needs_filling = true;
    } else {
      Y_needs_filling = false;
    }
  }
  void sample_Y_misaligned(const arma::uvec& theta_changed);
  
  // latent model 
  int latent_model; // 0: response, 1: block, 2: row seq, 3: col seq
  arma::mat W;
  void w_sequential_singlesite(const arma::uvec& theta_changed);
  void gibbs_w_sequential_byoutcome();
  void gibbs_w_block();
  void sample_Dvec();
  arma::vec Dvec;
  //
  
  // utility for latent model and misaligned response model
  arma::field<arma::mat> Rw_no_Q;
  arma::field<arma::mat> Pblanket_no_Q;
  void cache_blanket_comps(const arma::uvec& theta_changed);
  
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
  void gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq=false);
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
        const arma::mat& Beta_start,
        const arma::mat& Sigma_start,
        const arma::mat& daggp_theta, 
        const arma::uvec& update_theta_which,
        const arma::vec& tausq_start,
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
    
    manage_missing_data();
    
    if(latent_model>0){
      W = Y;
      W(arma::find_nonfinite(W)).fill(0);
      Dvec = tausq_start;
    }
    
    B = Beta_start;
    YXB = Y - X * B;
    
    B_Var = 1000 * arma::ones(arma::size(B));
    
    theta = daggp_theta;
    daggps = std::vector<DagGP>(q);
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = update_theta_which(0) == 1;
    sigmasq_sampling = update_theta_which(1) == 1;
    nu_sampling = update_theta_which(2) == 1;
    tausq_sampling = update_theta_which(3) == 1;
    
    matern = cov_model_matern;
    //Rcpp::Rcout << "Covariance choice: " << matern << endl;
    
    // make n_threads depend on whether data are gridded, since behavior is opposite
    
    int daggp_n_threads = dag_opts == -1? 1 : num_threads;
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag, 
                               dag_opts,
                               matern, 
                               latent_model==3, // with q blocks, make Ci
                               daggp_n_threads);
    }
    daggps_alt = daggps;
    
    init_theta_adapt();
    
    S = arma::chol(Sigma_start, "upper");
    Si = arma::inv(arma::trimatu(S));
    Sigma = S.t() * S;
    Q = Si * Si.t();
    
    compute_V();
    
    int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;
    Rw_no_Q = arma::field<arma::mat> (nfill);
    Pblanket_no_Q = arma::field<arma::mat> (nfill);
    
    arma::uvec updater = arma::ones<arma::uvec>(q);
    if(latent_model | Y_needs_filling){
      // first time making markov blanket cache
      cache_blanket_comps(updater);
    }
    
    timings = arma::zeros(10);
  }
};
