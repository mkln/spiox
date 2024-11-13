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
  double B_a_dl; // dirichlet-laplace parameter for vec(B)
  
  // SPF for sparse latent precision
  SparsePrecisionFactor spf;
  arma::mat S, Si, Sigma, Q; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // RadGP for spatial dependence
  std::vector<DagGP> daggp_options, daggp_options_alt;
  arma::mat theta_options; // each column is one alternative value for theta
  unsigned int n_options;
  arma::uvec spmap; // qx1 vector spmap(i) = which element of daggp_options for factor i
  
  arma::mat V; 
  
  // -------------- utilities
  bool matern;
  void sample_B(); // 
  void sample_Q_spf();
  void sample_Sigma_wishart();
  void compute_V(); // whitened
  void sample_theta_discr(); // gibbs for each outcome choosing from options
  void upd_theta_metrop();
  void upd_theta_metrop_conditional(); // n_options == q
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
  void gibbs(int it, int sample_sigma, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts);
  double logdens_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructor
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int latent_model_choice,
        const arma::mat& daggp_theta, 
        
        const arma::mat& Sigma_start,
        const arma::mat& mvreg_B_start,
        bool use_matern,
        int num_threads_in) 
  {
    num_threads = num_threads_in;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    latent_model = latent_model_choice; // latent model? 
    if(latent_model>0){
      W = Y;
      XtX = X.t() * X;
      Dvec = arma::ones(q)*.01;
    }
    
    B = mvreg_B_start;
    YXB = Y - X * B;
    
    B_Var = 1000 * arma::ones(arma::size(B));
    
    theta_options = daggp_theta;
    n_options = theta_options.n_cols;
    daggp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    
    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = arma::var(theta_options.row(0)) != 0;
    sigmasq_sampling = arma::var(theta_options.row(1)) != 0;
    nu_sampling = arma::var(theta_options.row(2)) != 0;
    tausq_sampling = arma::var(theta_options.row(3)) != 0;
    
    matern = use_matern;
    
    for(unsigned int i=0; i<n_options; i++){
      daggp_options[i] = DagGP(_coords, theta_options.col(i), custom_dag, //daggp_rho, 
                               matern, 
                               latent_model==3, // with q blocks, make Ci
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
    Sigma = S.t() * S;
    Q = Si * Si.t();
    
    compute_V();
    
    timings = arma::zeros(10);
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
  if(sigmasq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 1*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(tausq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = n_options * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({3, 100}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({0.4, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  bounds_all.row(3) = arma::rowvec({1e-6, 100}); // tausq
  bounds_all = bounds_all.rows(which_theta_elem);
  theta_unif_bounds = arma::zeros(0, 2);
  
  for(int j=0; j<n_options; j++){
    theta_unif_bounds = arma::join_vert(theta_unif_bounds, bounds_all);
  }
  
  arma::mat theta_metrop_sd = 0.05 * arma::eye(n_theta_par, n_theta_par);
  theta_adapt = RAMAdapt(n_theta_par, theta_metrop_sd, 0.24);
  theta_adapt_active = true;
  
  // conditional update
  c_theta_unif_bounds = bounds_all;
  int c_theta_par = which_theta_elem.n_elem;
  arma::mat c_theta_metrop_sd = 0.05 * arma::eye(c_theta_par, c_theta_par);
  c_theta_adapt.reserve(n_options);
  for(int j=0; j<n_options; j++){
    c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
  }
  // ---
}

inline void SpIOX::compute_V(){ 
  // V is made of
  // B (in the response model only)
  // theta, and cluster assignments
  
  // it is used in 
  // Sigma
  // gibbs for theta
  
  
  // whiten the residuals from spatial dependence
  if(latent_model>0){
    V = W;
  } else {
    V = YXB;
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int i=0; i<q; i++){
    V.col(i) = daggp_options.at(spmap(i)).H_times_A(V.col(i));
  }
}

inline void SpIOX::sample_B(){
  
  if(latent_model==0){
    // update B via gibbs for the response model
    //Rcpp::Rcout << "+++++++++++++++++ ORIG +++++++++++++++++++" << endl;
    //S^T * S = Sigma
    std::chrono::steady_clock::time_point tstart;
    std::chrono::steady_clock::time_point tend;
    int timed = 0;
    //Rcpp::Rcout << "------- builds0 ----" << endl;
    tstart = std::chrono::steady_clock::now();
    arma::mat Ytilde = Y;
    arma::mat Xtilde = arma::zeros(n*q, p*q);
    arma::vec daggp_logdets = arma::zeros(q);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j=0; j<q; j++){
      Ytilde.col(j) = daggp_options.at(spmap(j)).H_times_A(Y.col(j));// * Y.col(j);
      arma::mat HX = daggp_options.at(spmap(j)).H_times_A(X);// * X;
      daggp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
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
    
  } else {
    // update B via gibbs for the latent model
    // we could make this into a conjugate MN update rather than conj N
    arma::mat Ytilde = Y - W;
    arma::mat mvnorm = arma::randn(p, q);
    for(int i=0; i<q; i++){
      arma::mat post_precision = arma::diagmat(1.0/B_Var.col(i)) + XtX/Dvec(i);
      arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
      B.col(i) = pp_ichol.t() * (pp_ichol * X.t() * Ytilde.col(i)/Dvec(i) + mvnorm.col(i));
    }
  }
  
  YXB = Y - X*B;
  
}

inline void SpIOX::sample_theta_discr(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  arma::vec zz = arma::zeros(1);
  
  arma::mat Target;
  if(latent_model>0){
    Target = W;
  } else {
    Target = YXB; // Y - XB
  }
  
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
      V_loc.col(j) = daggp_options.at(r).H_times_A(Target.col(j));// * Ytilde.col(j);
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
    V.col(j) = daggp_options.at(spmap(j)).H_times_A(Target.col(j));// * Ytilde.col(j);
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
  
  if(!theta_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  // ---------------------
  // create proposal daggp
  // this can run in parallel but update_theta already uses omp
  // do not run this in parallel, will be faster this way
  tstart = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<n_options; i++){
    daggp_options_alt[i].update_theta(theta_alt.col(i), true);
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();

  // ----------------------
  // current density and proposal density
  tstart = std::chrono::steady_clock::now();
  
  arma::vec daggp_logdets = arma::zeros(q);
  arma::mat V_alt = V;
  arma::vec daggp_alt_logdets = arma::zeros(q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    //vecYtilde.subvec(j*n, (j+1)*n-1) = daggp_options.at(spmap(j)).H * vecYtilde.subvec(j*n, (j+1)*n-1);
    //vecYtilde_alt.subvec(j*n, (j+1)*n-1) = daggp_options_alt.at(spmap(j)).H * vecYtilde_alt.subvec(j*n, (j+1)*n-1);
    arma::mat Target;
    if(latent_model>0){
      Target = W.col(j);
    } else {
      Target = YXB.col(j);
    }
    V_alt.col(j) = daggp_options_alt.at(spmap(j)).H_times_A(Target);// * (Y.col(j) - X * B.col(j));
    daggp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
    daggp_alt_logdets(j) = daggp_options_alt.at(spmap(j)).precision_logdeterminant;
  }
  
  // current
  //arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  //arma::vec ytilde = arma::vectorise(V * Si);
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(V * Si, 2.0));
  
  // proposal
  //arma::mat Ytildemat_alt = arma::mat(vecYtilde_alt.memptr(), n, q, false, true);
  //arma::vec ytilde_alt = arma::vectorise(V_alt * Si);
  double prop_ldet = +0.5*arma::accu(daggp_alt_logdets);
  double prop_logdens = prop_ldet - 0.5*arma::accu(pow(V_alt * Si, 2.0));
  
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
  // priors
  double logpriors = 0;
  for(unsigned int j=0; j<n_options; j++){
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta_options(1,j), 2, 1);
    }
    if(tausq_sampling){
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta_options(3,j), 25);
    }
  }
  
  // ------------------
  // make move
  double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
  
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

inline void SpIOX::upd_theta_metrop_conditional(){
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int j=0; j<n_options; j++){
    c_theta_adapt[j].count_proposal();
    
    arma::vec phisig_cur = theta_options(which_theta_elem, oneuv*j);
    
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(phisig_cur.n_elem);
    
    arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
      phisig_cur, theta_unif_bounds) + 
        c_theta_adapt[j].paramsd * U_update, theta_unif_bounds);
    
    // proposal for theta matrix
    arma::mat theta_alt = theta_options;
    theta_alt(which_theta_elem, oneuv*j) = phisig_alt; 
    
    if(!theta_alt.is_finite()){
      Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
    }
    
    // ---------------------
    // create proposal daggp
    daggp_options_alt[j].update_theta(theta_alt.col(j), true);
    
    // conditional density of Y_j | Y_-j (or W depending on target)
    arma::mat V_alt = V;
    if(latent_model>0){
      V_alt.col(j) = daggp_options_alt.at(spmap(j)).H_times_A(W.col(j));// * (Y.col(j) - X * B.col(j));
    } else {
      V_alt.col(j) = daggp_options_alt.at(spmap(j)).H_times_A(YXB.col(j));// * (Y.col(j) - X * B.col(j));
    }
    
    double c_daggp_logdet = daggp_options.at(spmap(j)).precision_logdeterminant;
    double c_daggp_alt_logdet = daggp_options_alt.at(spmap(j)).precision_logdeterminant;
    
    arma::vec Vjc = arma::zeros(n);
    arma::vec Vjc_alt = arma::zeros(n);
    for(int jc=0; jc<q; jc++){
      if(jc != j){
        Vjc += Q(j, jc)/Q(j,j) * V.col(jc);
        Vjc_alt += Q(j, jc)/Q(j,j) * V_alt.col(jc);
      }
    }
    double core_alt = arma::accu(pow(V_alt.col(j) + Vjc_alt, 2.0)); 
    double core = arma::accu(pow(V.col(j) + Vjc, 2.0)); 
    double prop_logdens = 0.5 * c_daggp_alt_logdet - 0.5 * core_alt;
    double curr_logdens = 0.5 * c_daggp_logdet - 0.5 * core;
    
    // priors
    double logpriors = 0;
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta_options(1,j), 2, 1);
    }
    if(tausq_sampling){
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta_options(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    bool accepted = do_I_accept(logaccept);
    
    if(accepted){
      theta_options = theta_alt;
      std::swap(daggp_options, daggp_options_alt);
      std::swap(V, V_alt);
    } 
    
    c_theta_adapt[j].update_ratios();
    
    if(theta_adapt_active){
      c_theta_adapt[j].adapt(U_update, exp(logaccept), theta_mcmc_counter); 
    }
    
    theta_mcmc_counter++;
    
    
  }
  
}

inline void SpIOX::gibbs_w_sequential_singlesite(){
  // stuff to be moved to SpIOX class for latent model
  arma::mat Di = arma::diagmat(1/Dvec);
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  arma::mat mvnorm = arma::randn(q, n);
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  arma::field<arma::mat> Ctchol(n);
  
  // V = whitened Y-XB or W
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggp_options[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    
    Rw(i) = arma::zeros(q, q); 
    arma::mat Pblanket(q, q*mbsize);
    for(int r=0; r<q; r++){
      for(int s=0; s<q; s++){
        Rw(i)(r, s) = Q(r,s) * 
          arma::accu( daggp_options[spmap(r)].H.col(i) % 
          daggp_options[spmap(s)].H.col(i) );
        
        int startcol = s * mbsize;
        int endcol = (s + 1) * mbsize - 1;
        for(int j = 0; j < mblanket.n_elem; j++) {
          int col_idx = mblanket(j);
          Pblanket(r, startcol + j) = Q(r, s) *
            arma::accu(daggp_options[spmap(r)].H.col(i) %
                      daggp_options[spmap(s)].H.col(col_idx));
        }
      }
    }
  
    Hw(i) = - Pblanket;
    Ctchol(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di, "lower")));
  }

  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int i=0; i<n; i++){
    arma::uvec mblanket = daggp_options[0].mblanket(i);
    arma::vec meancomp = Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di * arma::trans(YXB.row(i)); 
    
    W.row(i) = arma::trans( Ctchol(i).t() * (Ctchol(i) * meancomp + mvnorm.col(i) ));
  }
  
  for(unsigned int i=0; i<q; i++){
    W.col(i) = W.col(i) - arma::mean(W.col(i));
  }
}

inline void SpIOX::gibbs_w_sequential_byoutcome(){

  std::vector<arma::sp_mat> prior_precs(q);
  std::vector<arma::spsolve_factoriser> factoriser(q);
  arma::uvec statuses = arma::ones<arma::uvec>(q);
  
  arma::superlu_opts opts;
  opts.symmetric  = true;
  
  arma::mat urands = arma::randn(n, q);
  arma::mat vrands = arma::randn(n, q);
  
  arma::mat HDs = arma::zeros(n, q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int j=0; j<q; j++){
    // compute prior precision
    arma::sp_mat post_prec = Q(j,j) * daggp_options[spmap(j)].Ci; //daggp_options[spmap(j)].H.t() * daggp_options[spmap(j)].H;
    // compute posterior precision
    post_prec.diag() += 1.0/Dvec(j);
    
    // factorise precision so we can use it for sampling
    bool status = factoriser[j].factorise(post_prec, opts);
    if(status == false) { statuses(j) = 0; }
    
    // this does not need sequential
    HDs.col(j) = YXB.col(j)/Dvec(j) + 
      urands.col(j)/sqrt(Dvec(j)) + 
      sqrt(Q(j,j)) * daggp_options[spmap(j)].H.t() * vrands.col(j);
  }
  
  if(arma::any(statuses==0)){ Rcpp::stop("Failed factorization for sampling w."); }
  
  arma::uvec r1q = arma::regspace<arma::uvec>(0,q-1);
  for(int j=0; j<q; j++){
    arma::vec x = arma::zeros(n);
    arma::uvec notj = arma::find(r1q != j);
    arma::uvec jx = arma::zeros<arma::uvec>(1) + j;
    // V is whitened W as we want
    arma::vec Mi_m_prior = - daggp_options[spmap(j)].H.t() * V.cols(notj) * Q.submat(notj, jx);
    arma::vec rhs = Mi_m_prior + HDs.col(j);
    
    arma::vec w_sampled; 
    bool foundsol = factoriser[j].solve(w_sampled, rhs);
    
    W.col(j) = w_sampled - arma::mean(w_sampled);
    
    V.col(j) = daggp_options.at(spmap(j)).H_times_A(W.col(j));
  }
  
}

inline void SpIOX::gibbs_w_block(){
  // ------------------ slow 
  
  if(q*n > 5000){
    Rcpp::stop("Use another sampler. This block sampler was not coded for data this size.\n");
  }
  
  // precision matrix via hadamard product
  arma::sp_mat Hvert(q*n, n);
  
  // Offset for placing each matrix
  arma::uword row_offset = 0;
  arma::uword col_offset = 0;
  
  arma::mat Unorm = arma::randn(n,q) * Si.t();
  
  
  for (int i=0; i<q; i++) {
    // Insert the matrix at the correct block position
    Hvert.submat(row_offset, 0, row_offset + n - 1, n - 1) = 
      daggp_options[spmap(i)].H.t();
    row_offset += n;
    
    Unorm.col(i) = daggp_options[spmap(i)].H.t() * Unorm.col(i);
  }
  // U = {+Li^T} (Si %x% In) * rnorm(nq)
  
  arma::sp_mat post_prec = Hvert*Hvert.t();
  for (arma::sp_mat::iterator it = post_prec.begin(); it != post_prec.end(); ++it) {
    int r = it.row();  // Row index of the nonzero element
    int s = it.col();  // Column index of the nonzero element
    int i = r / n;  
    int j = s / n;  
    post_prec(r, s) = Q(i, j) * post_prec(r, s);
    if(r == s){
      post_prec(r, s) += 1/Dvec(i);
    }
  }
  
  arma::mat Di = arma::diagmat(1/Dvec);
  arma::vec post_Cmean = arma::vectorise(YXB * Di);
  arma::vec vnorm = arma::vectorise( arma::randn(n, q) * arma::diagmat(sqrt(1.0/Dvec)) );
  
  arma::vec post_meansample = post_Cmean + arma::vectorise(Unorm) + vnorm;
  
  arma::superlu_opts opts;
  opts.symmetric  = true;
  
  arma::spsolve_factoriser factr;
  bool okfact = factr.factorise(post_prec, opts);
  
  if(okfact == false){ Rcpp::stop("Failed factorization for sampling w."); }
  
  arma::vec w;
  bool foundsol = factr.solve(w, post_meansample);
  
  if(foundsol == false)  { Rcpp::stop("Could not solve for sampling w."); }
  W = arma::mat(w.memptr(), n, q);
  
  for(unsigned int i=0; i<q; i++){
    W.col(i) = W.col(i) - arma::mean(W.col(i));
  }
}

inline void SpIOX::sample_Dvec(){
  for(int i=0; i<q; i++){
    double ssq = arma::accu(pow( YXB.col(i) - W.col(i), 2));
    Dvec(i) = 1.0/R::rgamma(n/2 + 1e-5, 1.0/(1e-5 + 0.5 * ssq));
  }
}

inline void SpIOX::sample_Q_spf(){
  spf.replace_Y(&V);
  spf.fc_sample_uv();
  spf.fc_sample_Lambda();
  spf.fc_sample_Delta();
  spf.fc_sample_dl();
  
  // compute other stuff
  
  // cholesky low rank update function
  arma::mat U = arma::diagmat(sqrt(spf.Delta));
  uchol_update(U, spf.Lambda);
  
  Si = U.t();
  S = arma::inv(arma::trimatl(Si));
  
  Sigma = S.t() * S;
  Q = Si * Si.t();
}

inline void SpIOX::sample_Sigma_wishart(){
  arma::mat Smean = n * arma::cov(V) + arma::eye(V.n_cols, V.n_cols);
  arma::mat Q_mean_post = arma::inv_sympd(Smean);
  double df_post = n + (V.n_cols);
  
  Q = arma::wishrnd(Q_mean_post, df_post);
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  Sigma = S.t() * S;
}

inline void SpIOX::gibbs(int it, int sample_sigma, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts){
  
  if(sample_sigma > 0){
    if(sample_sigma == 1){
      tstart = std::chrono::steady_clock::now();
      sample_Q_spf();
      timings(2) += time_count(tstart); 
    } else {
      tstart = std::chrono::steady_clock::now();
      sample_Sigma_wishart();
      timings(2) += time_count(tstart); 
    }
  }
  
  if(sample_mvr){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    sample_B();
    timings(0) += time_count(tstart);  
    
    // V = whitened Y-XB or W
    tstart = std::chrono::steady_clock::now();
    compute_V();
    timings(1) += time_count(tstart);
  }
  
  // update theta | Y, S based on discrete prior
  tstart = std::chrono::steady_clock::now();
  if(sample_theta_gibbs){
    sample_theta_discr();
  }
  timings(3) += time_count(tstart);
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  if(upd_theta_opts){
    if(n_options == q){
      upd_theta_metrop_conditional();
    } else {
      upd_theta_metrop();
    }
  }
  timings(4) += time_count(tstart);
  
  if(latent_model > 0){
    tstart = std::chrono::steady_clock::now();
    if(latent_model == 1){
      gibbs_w_block();
      compute_V();
    } 
    if(latent_model == 2){
      gibbs_w_sequential_singlesite();
      compute_V();
    }
    if(latent_model == 3){
      gibbs_w_sequential_byoutcome();
    }
    sample_Dvec();
    timings(5) += time_count(tstart);
  }
}

inline double SpIOX::logdens_eval(){
  arma::vec vecYtilde = arma::vectorise(Y - X * B);
  arma::vec daggp_logdets = arma::zeros(q);
  for(unsigned int j=0; j<q; j++){
    arma::mat vYtlocal = vecYtilde.subvec(j*n, (j+1)*n-1);
    vecYtilde.subvec(j*n, (j+1)*n-1) = daggp_options.at(spmap(j)).H_times_A(vYtlocal);// * vecYtilde.subvec(j*n, (j+1)*n-1);
    daggp_logdets(j) = daggp_options.at(spmap(j)).precision_logdeterminant;
  }
  
  arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  arma::vec ytilde = arma::vectorise(Ytildemat * Si);
  
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(ytilde, 2.0));
  
  return curr_logdens;
}

