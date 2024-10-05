#include "omp_import.h"
#include "daggp.h"
#include "ramadapt.h"
#include "spf.h"
#include "cholesky_lrupd.h"

using namespace std;

inline int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

class SpIOX {
public:
  // response:
  bool model_response;
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
  arma::mat B_Var; // prior variance on B, element by element
  double B_a_dl; // dirichlet-laplace parameter for vec(B)
  
  // SPF for sparse latent precision
  SparsePrecisionFactor spf;
  arma::mat S, Si; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // RadGP for spatial dependence
  std::vector<DagGP> daggp_options, daggp_options_alt;
  arma::mat theta_options; // each column is one alternative value for theta
  unsigned int n_options;
  arma::uvec spmap; // qx1 vector spmap(i) = which element of daggp_options for factor i
  
  arma::mat V; 
  
  // -------------- utilities
  void sample_B(); // 
  void compute_V(); // whitened
  void compute_S();
  void sample_theta_discr(); // gibbs for each outcome choosing from options
  void upd_theta_metrop();
  void init_theta_adapt();
  
  bool phi_sampling, nu_sampling, tausq_sampling;
  // adaptive metropolis to update theta atoms
  int theta_mcmc_counter;
  arma::uvec which_theta_elem;
  arma::mat theta_unif_bounds;
  arma::mat theta_metrop_sd;
  RAMAdapt theta_adapt;
  bool theta_adapt_active;
  // --------
  
  // -------------- run 1 gibbs iteration based on current values
  void gibbs_response(int it, int sample_precision, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts);
  double logdens_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructors
  // for response model inverse Wishart
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        //double radgp_rho, 
        const arma::mat& radgp_theta, 
        
        const arma::mat& Sigma_start,
        const arma::mat& mvreg_B_start,
        int num_threads_in) 
  {
    model_response = true;
    num_threads = num_threads_in;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    B = mvreg_B_start;
    B_Var = arma::ones(arma::size(B));
    
    theta_options = radgp_theta;
    n_options = theta_options.n_cols;
    daggp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = arma::var(theta_options.row(0)) != 0;
    nu_sampling = arma::var(theta_options.row(2)) != 0;
    tausq_sampling = arma::var(theta_options.row(3)) != 0;
    
    for(unsigned int i=0; i<n_options; i++){
      daggp_options[i] = DagGP(_coords, theta_options.col(i), custom_dag, //radgp_rho, 
                               nu_sampling, //matern 
                               num_threads);
    }
    daggp_options_alt = daggp_options;
    if(n_options < q){
      // will need update from discrete
      spmap = arma::zeros<arma::uvec>(q);
    } else { 
      // 
      spmap = arma::regspace<arma::uvec>(0, q-1);
    }
    
    init_theta_adapt();
    
    S = arma::chol(Sigma_start, "upper");
    Si = arma::inv(arma::trimatu(S));
    
    compute_V();
    
    timings = arma::zeros(10);
  }
  
  
  // for response model spf
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        
        double radgp_rho, 
        const arma::mat& radgp_theta, 
        
        int spf_k, double spf_a_delta, double spf_b_delta, double spf_a_dl,
        
        const arma::mat& spf_Lambda_start, const arma::vec& spf_Delta_start,
        const arma::mat& mvreg_B_start,
        int num_threads_in) 
  {
    model_response = true;
    num_threads = num_threads_in;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    B = mvreg_B_start;
    B_Var = 100*arma::ones(arma::size(B));
    
    theta_options = radgp_theta;
    n_options = theta_options.n_cols;
    daggp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    
    phi_sampling = arma::var(theta_options.row(0)) != 0;
    nu_sampling = arma::var(theta_options.row(2)) != 0;
    tausq_sampling = arma::var(theta_options.row(3)) != 0;
    
    for(unsigned int i=0; i<n_options; i++){
      daggp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, nu_sampling, 
                               num_threads);
    }
    daggp_options_alt = daggp_options;
    if(n_options < q){
      // will need update from discrete
      spmap = arma::zeros<arma::uvec>(q);
    } else { 
      // 
      spmap = arma::regspace<arma::uvec>(0, q-1);
    }
    
    init_theta_adapt();
    
    spf = SparsePrecisionFactor(&Y, spf_k, spf_a_delta, spf_b_delta, spf_a_dl);
    spf.Lambda_start(spf_Lambda_start);
    spf.Delta_start(spf_Delta_start);
    
    compute_S();
    compute_V();
    
    timings = arma::zeros(10);
  }
  
  // for predictions
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        
        double radgp_rho, const arma::mat& radgp_theta) 
  {
    model_response = true;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    theta_options = radgp_theta;
    n_options = theta_options.n_cols;
    daggp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    for(unsigned int i=0; i<n_options; i++){
      daggp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1);
    }
  }  
  
};

inline void SpIOX::init_theta_adapt(){
  // adaptive metropolis
  theta_mcmc_counter = 0;
  which_theta_elem = arma::zeros<arma::uvec>(0);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  if(phi_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 0*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(tausq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = n_options * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({0.1, 100});
  bounds_all.row(1) = arma::rowvec({1e-6, 100});
  bounds_all.row(2) = arma::rowvec({0.4, 2});
  bounds_all.row(3) = arma::rowvec({1e-6, 100});
  bounds_all = bounds_all.rows(which_theta_elem);
  theta_unif_bounds = arma::zeros(0, 2);
  
  for(int j=0; j<n_options; j++){
    theta_unif_bounds = arma::join_vert(theta_unif_bounds, bounds_all);
  }
  
  theta_metrop_sd = 0.05 * arma::eye(n_theta_par, n_theta_par);
  theta_adapt = RAMAdapt(n_theta_par, theta_metrop_sd, 0.24);
  theta_adapt_active = true;
  // ---
}

inline void SpIOX::compute_V(){
  // whiten the residuals from spatial dependence
  V = ( Y - X * B );
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int i=0; i<q; i++){
    V.col(i) = //daggp_options.at(spmap(i)).H * V.col(i);  
      daggp_options.at(spmap(i)).H_times_A(V.col(i));
  }
  // V = Li * (y-XB)
}

inline void SpIOX::compute_S(){
  //arma::mat Q = spf.Lambda * spf.Lambda.t() + arma::diagmat(spf.Delta);
  //Si = arma::trimatl(arma::chol(Q, "lower"));
  
  // cholesky low rank update function
  arma::mat U = arma::diagmat(sqrt(spf.Delta));
  uchol_update(U, spf.Lambda);
  
  Si = U.t();
  S = arma::inv(arma::trimatl(Si));
  // Sigma = S^T * S = (Si * Si^T)^-1
}

inline void SpIOX::sample_B(){
  //Rcpp::Rcout << "+++++++++++++++++ ORIG +++++++++++++++++++" << endl;
  //S^T * S = Sigma
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  //Rcpp::Rcout << "------- builds0 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  arma::mat Ytilde = Y;
  arma::mat Xtilde = arma::zeros(n*q, p*q);
  arma::vec radgp_logdets = arma::zeros(q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = daggp_options.at(spmap(j)).H_times_A(Y.col(j));// * Y.col(j);
    arma::mat HX = daggp_options.at(spmap(j)).H_times_A(X);// * X;
    radgp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX; 
    }
  }
  
  arma::vec ytilde = arma::vectorise(Ytilde * Si);
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
  //Rcpp::Rcout << "------- builds3 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  
  arma::vec vecB_Var = arma::vectorise(B_Var);
  arma::mat post_precision = arma::diagmat(1.0/vecB_Var) + Xtilde.t() * Xtilde;
  arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
  arma::vec beta = pp_ichol.t() * (pp_ichol * Xtilde.t() * ytilde + arma::randn(p*q));
  B = arma::mat(beta.memptr(), p, q);
  
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
}

inline void SpIOX::sample_theta_discr(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  arma::vec zz = arma::zeros(1);
  
  arma::mat Ytilde = Y - X*B;
  
  // loop over outcomes -- this is gibbs so cannot parallelize
  for(unsigned int j=0; j<q; j++){
    arma::vec opt_logdens = arma::zeros(n_options);
    // -- loop over options // parallel ok if we create a copy of ytilde and Xtilde
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int r=0; r<n_options; r++){
      // compute unnorm logdens 
      // change option for this outcome: spmap(i) is the new theta option
      arma::mat V_loc = V;
      V_loc.col(j) = daggp_options.at(r).H_times_A(Ytilde.col(j), true);// * Ytilde.col(j);
      double opt_prec_logdet = daggp_options.at(r).precision_logdeterminant;
      
      // at each i, we ytilde and Xtilde remain the same except for outcome j
      opt_logdens(r) = 0.5 * opt_prec_logdet - 0.5 * arma::accu(pow(V_loc * Si, 2.0));
    }
    // these probabilities are unnormalized
    //Rcpp::Rcout << "options probs unnorm: " << opt_logdens.t() << endl;
    
    double c = arma::max(opt_logdens);
    double log_norm_const = c + log(arma::accu(exp(opt_logdens - c)));
    // now normalize
    opt_logdens = exp(opt_logdens - log_norm_const);
    //Rcpp::Rcout << "outcome j " << j << " " << opt_logdens.t() << endl;
    
    // finally sample
    double u = arma::randu();
    arma::vec cprobs = arma::join_vert(zz, arma::cumsum(opt_logdens));
    // reassign process hyperparameters based on gp density
    spmap(j) = arma::max(arma::find(cprobs < u));
    
    // update ytilde and Xtilde accordingly so we can move on to the next
    V.col(j) = daggp_options.at(spmap(j)).H_times_A(Ytilde.col(j));// * Ytilde.col(j);
    //Rcpp::Rcout << "done" << endl;
  }
  
}

inline void SpIOX::upd_theta_metrop(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  theta_adapt.count_proposal();
  
  arma::vec phisig_cur = arma::vectorise( theta_options.rows(which_theta_elem) );
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(phisig_cur.n_elem);
  
  arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
    phisig_cur, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  arma::mat theta_alt = theta_options;
  arma::mat phisig_alt_mat = arma::mat(phisig_alt.memptr(), which_theta_elem.n_elem, n_options);
  
  theta_alt.rows(which_theta_elem) = phisig_alt_mat; 
  
  // ---------------------
  // create proposal radgp
  //Rcpp::Rcout << "------- builds1 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<n_options; i++){
    daggp_options_alt[i].update_theta(theta_alt.col(i));
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
  // ----------------------
  // current density and proposal density
  tstart = std::chrono::steady_clock::now();
  
  //arma::vec vecYtilde = arma::vectorise(Y - X * B);
  arma::vec radgp_logdets = arma::zeros(q);
  //arma::vec vecYtilde_alt = vecYtilde;
  arma::mat V_alt = V;
  arma::vec radgp_alt_logdets = arma::zeros(q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    //vecYtilde.subvec(j*n, (j+1)*n-1) = daggp_options.at(spmap(j)).H * vecYtilde.subvec(j*n, (j+1)*n-1);
    //vecYtilde_alt.subvec(j*n, (j+1)*n-1) = daggp_options_alt.at(spmap(j)).H * vecYtilde_alt.subvec(j*n, (j+1)*n-1);
    arma::mat YXBj = Y.col(j) - X*B.col(j);
    V_alt.col(j) = daggp_options_alt.at(spmap(j)).H_times_A(YXBj);// * (Y.col(j) - X * B.col(j));
    radgp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
    radgp_alt_logdets(j) = daggp_options_alt.at(spmap(j)).precision_logdeterminant;
  }
  
  // current
  //arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  //arma::vec ytilde = arma::vectorise(V * Si);
  double curr_ldet = +0.5*arma::accu(radgp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(V * Si, 2.0));
  
  // proposal
  //arma::mat Ytildemat_alt = arma::mat(vecYtilde_alt.memptr(), n, q, false, true);
  //arma::vec ytilde_alt = arma::vectorise(V_alt * Si);
  double prop_ldet = +0.5*arma::accu(radgp_alt_logdets);
  double prop_logdens = prop_ldet - 0.5*arma::accu(pow(V_alt * Si, 2.0));
  
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
  // ------------------
  // make move
  double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + jacobian;
  bool accepted = do_I_accept(logaccept);
  
  if(accepted){
    theta_options = theta_alt;
    std::swap(daggp_options, daggp_options_alt);
    std::swap(V, V_alt);
  } 
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
  }
  
  theta_mcmc_counter++;
  
  //Rcpp::Rcout << theta_options.row(0) << endl;
  //Rcpp::Rcout << spmap.t() << endl;
}


inline void SpIOX::gibbs_response(int it, int sample_precision, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts){
  
  if(sample_precision > 0){
    if(sample_precision == 1){
      //Rcpp::Rcout << "L " << endl;
      // sample explicit precision factor loadings Lambda and 'precision noise' Delta
      tstart = std::chrono::steady_clock::now();
      spf.replace_Y(&V);
      spf.fc_sample_uv();
      spf.fc_sample_Lambda();
      spf.fc_sample_Delta();
      spf.fc_sample_dl();
      timings(2) += time_count(tstart); 
      
      
      //Rcpp::Rcout << "S " << endl;
      // compute S: implicit covariance factor loadings
      tstart = std::chrono::steady_clock::now();
      compute_S();
      timings(3) += time_count(tstart);
      
    } else {
      //Rcpp::Rcout << "W" << endl;
      tstart = std::chrono::steady_clock::now();
      arma::mat Smean = n * arma::cov(V) + arma::eye(V.n_cols, V.n_cols);
      arma::mat Q_mean_post = arma::inv_sympd(Smean);
      double df_post = n + (V.n_cols);
      
      arma::mat Q = arma::wishrnd(Q_mean_post, df_post);
      
      Si = arma::chol(Q, "lower");
      S = arma::inv(arma::trimatl(Si));
      
      timings(2) += time_count(tstart); 
    }
  }
  
  if(sample_mvr){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    sample_B();
    timings(0) += time_count(tstart);  
  }
  
  tstart = std::chrono::steady_clock::now();
  compute_V();
  timings(1) += time_count(tstart);
  
  //Rcpp::Rcout << "T " << endl;
  // update theta | Y, S based on discrete prior
  tstart = std::chrono::steady_clock::now();
  if(sample_theta_gibbs){
    sample_theta_discr();
  }
  timings(4) += time_count(tstart);
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  if(upd_theta_opts){
    upd_theta_metrop();
  }
  timings(5) += time_count(tstart);
  
  
}

inline double SpIOX::logdens_eval(){
  arma::vec vecYtilde = arma::vectorise(Y - X * B);
  arma::vec radgp_logdets = arma::zeros(q);
  for(unsigned int j=0; j<q; j++){
    arma::mat vYtlocal = vecYtilde.subvec(j*n, (j+1)*n-1);
    vecYtilde.subvec(j*n, (j+1)*n-1) = daggp_options.at(spmap(j)).H_times_A(vYtlocal);// * vecYtilde.subvec(j*n, (j+1)*n-1);
    radgp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
  }
  
  arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  arma::vec ytilde = arma::vectorise(Ytildemat * Si);
  
  double curr_ldet = +0.5*arma::accu(radgp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(ytilde, 2.0));
  
  return curr_logdens;
}

