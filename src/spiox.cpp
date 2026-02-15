#include "spiox.h"

using namespace std;


int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

void SpIOX::update_running_means(arma::mat& E_A, arma::mat& A){
  double min_iterations = 100.0;
  double gamma = 0.7;
  
  double alpha = vi_it < min_iterations ? 1 : 1.0/pow(vi_it - min_iterations + 1, gamma); // weight for present
  E_A = (1-alpha) * E_A + alpha * A;
  
  // vi_it is the internal iteration counter
}

void SpIOX::init_theta_adapt(){
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
  if(alpha_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = q * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 100}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({1e-5, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  //bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
  // 1-alpha is the proportion of variance explained by spatial component
  // alpha is is the proportion explained by nugget effect
  bounds_all.row(3) = arma::rowvec({1e-10, 1-1e-10}); // alpha
  bounds_all = bounds_all.rows(which_theta_elem);
  theta_unif_bounds = arma::zeros(0, 2);
  
  for(int j=0; j<q; j++){
    theta_unif_bounds = arma::join_vert(theta_unif_bounds, bounds_all);
  }
  
  arma::mat theta_metrop_sd = 0.05 * arma::eye(n_theta_par, n_theta_par);
  theta_adapt = RAMAdapt(n_theta_par, theta_metrop_sd, 0.24);
  theta_adapt_active = true;
  
  if(q > 2){
    // conditional update for more than 2 outcomes
    c_theta_unif_bounds = bounds_all;
    int c_theta_par = which_theta_elem.n_elem;
    arma::mat c_theta_metrop_sd = 0.05 * arma::eye(c_theta_par, c_theta_par);
    c_theta_adapt = std::vector<RAMAdapt>(q);
    for(int j=0; j<q; j++){
      c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
    }
    // ---  
  }
  
}

void SpIOX::compute_V(){ 
  // V is made of
  // B (in the response model only)
  // theta, and cluster assignments
  
  // it is used in 
  // Sigma
  // gibbs for theta
  
  
  // whiten the residuals from spatial dependence
  if(latent_model>0){
    V = W;
    //if(vi){
    //  V_samples_vi = W_samples_vi;
    //}
    
  } else {
    V = YXB;
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    V.col(j) = daggps.at(j).H_times_A(V.col(j));
  }
  
  bool do_VTV = vi & (latent_model>0);
  if(do_VTV){
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(num_threads)
//#endif
//    for(int ss=0; ss<N_mcvi_samples; ss++){
//      for(int j=0; j<q; j++){
//        arma::vec Vtemp = V_samples_vi.subcube(0, j, ss, n-1, j, ss);
//        V_samples_vi.subcube(0, j, ss, n-1, j, ss) = daggps.at(j).H_times_A(Vtemp);
//      }
//    }
    
//    VTV = arma::zeros(q, q);
//    for(int ss=0; ss<N_mcvi_samples; ss++){
//      arma::mat Vtemp = V_samples_vi.slice(ss);
    VTV = V.t() * V; //1.0/N_mcvi_samples * Vtemp.t() * Vtemp;
//    }
    update_running_means(VTV_ma, VTV);
  
  }
  
  
  
}

void SpIOX::update_B(){
  if(latent_model==0){
    // update B via gibbs for the response model
    
    //S^T * S = Sigma
    std::chrono::steady_clock::time_point tstart;
    std::chrono::steady_clock::time_point tend;
    int timed = 0;
    //Rcpp::Rcout << "------- builds0 ----" << endl;
    tstart = std::chrono::steady_clock::now();
    arma::mat Ytilde = Y;
    
    arma::vec daggp_logdets = arma::zeros(q);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j=0; j<q; j++){
      Ytilde.col(j) = daggps.at(j).H_times_A(Y.col(j));// * Y.col(j);
      daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
      
      if(HX_needs_updating){
        HX.slice(j) = daggps.at(j).H_times_A(X);// * X;
      }
      
      
      for(unsigned int i=0; i<q; i++){
        Xtilde.submat(i * n,       j * p,
                      (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX.slice(j); 
      }
    }
    
    if(HX_needs_updating){
      // we have updated here, switch off
      HX_needs_updating = false;
    }
    
    arma::vec ytilde = arma::vectorise(Ytilde * Si);
    tend = std::chrono::steady_clock::now();
    timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    //Rcpp::Rcout << timed << endl;
    
    //Rcpp::Rcout << "------- builds3 ----" << endl;
    tstart = std::chrono::steady_clock::now();
    
    arma::mat randnormat = (vi ? 0.0 : 1.0) * arma::randn(p*q);
    
    arma::vec vecB_Var = arma::vectorise(B_Var);
    arma::mat post_precision = arma::diagmat(1.0/vecB_Var) + Xtilde.t() * Xtilde;
    arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
    Beta_UQ = pp_ichol.t() * pp_ichol;
    arma::vec beta = Beta_UQ * Xtilde.t() * ytilde + pp_ichol.t() *randnormat;
    B = arma::mat(beta.memptr(), p, q);
    
    tend = std::chrono::steady_clock::now();
    timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    //Rcpp::Rcout << timed << endl;
    
  } else {
    // update B via gibbs for the latent model
    // btw we could make this into a conjugate MN update rather than conj N
    arma::mat Ytilde = Y - W;
    arma::mat mvnorm = arma::randn(p, q);
    for(int j=0; j<q; j++){
      arma::mat X_available = X.rows(avail_by_outcome(j));
      arma::mat XtX = X_available.t() * X_available;
      arma::mat post_precision = arma::diagmat(1.0/B_Var.col(j)) + XtX/Dvec(j);
      arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
      arma::vec yj = Ytilde.col(j);
      arma::vec y_available = yj.rows(avail_by_outcome(j));
      arma::vec XtYtildej = arma::trans(X_available) * y_available;
      
      arma::vec B_mean = pp_ichol.t() * pp_ichol * XtYtildej/Dvec(j);
      B.col(j) = B_mean + pp_ichol.t() * mvnorm.col(j);
      
    }
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::update_BW_asis(arma::mat& B, arma::mat& W, bool sampling){
  if(latent_model>0){
    // update B via gibbs for the response model
    //Rcpp::Rcout << "+++++++++++++++++ ORIG +++++++++++++++++++" << endl;
    //S^T * S = Sigma
    std::chrono::steady_clock::time_point tstart;
    std::chrono::steady_clock::time_point tend;
    int timed = 0;
    
    //
    arma::mat eta = X * B + W;
    
    //Rcpp::Rcout << " asis eta = XB + W " << endl;
    tstart = std::chrono::steady_clock::now();
    
    arma::mat Ytilde = eta; // we will whiten this in the loop below
    //Xtilde = arma::zeros(n*q, p*q);
    arma::vec daggp_logdets = arma::zeros(q);
    // whitening of eta and X
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j=0; j<q; j++){
      Ytilde.col(j) = daggps.at(j).H_times_A(Y.col(j));// * Y.col(j);
      daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
      
      if(HX_needs_updating){
        HX.slice(j) = daggps.at(j).H_times_A(X);// * X;
      }
  
      for(unsigned int i=0; i<q; i++){
        Xtilde.submat(i * n,       j * p,
                      (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX.slice(j); 
      }
    }
    arma::vec ytilde = arma::vectorise(Ytilde * Si);
    tend = std::chrono::steady_clock::now();
    timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    
    if(HX_needs_updating){
      // we have updated here, switch off
      HX_needs_updating = false;
    }
    
    //Rcpp::Rcout << "------- builds3 ----" << endl;
    tstart = std::chrono::steady_clock::now();
    
    arma::mat randnormat = (sampling? 1.0 : 0.0) * arma::randn(p*q);
    
    arma::vec vecB_Var = arma::vectorise(B_Var);
    arma::mat post_precision = arma::diagmat(1.0/vecB_Var) + Xtilde.t() * Xtilde;
    arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
    arma::vec beta = pp_ichol.t() * (pp_ichol * Xtilde.t() * ytilde + randnormat);
    B = arma::mat(beta.memptr(), p, q);
    
    W = eta - X*B;
    
    tend = std::chrono::steady_clock::now();
    timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    //Rcpp::Rcout << timed << endl;
  } 
  
}

void SpIOX::vi_Beta_UQ(){
  arma::vec beta_v = arma::vectorise(B);
  
  delta_t = beta_v - beta_running_mean;
  beta_running_mean = beta_running_mean + 1.0/(1.0+vi_it) * delta_t;
  Beta_UQ = Beta_UQ + delta_t * arma::trans(beta_v - beta_running_mean);
}

/*
void SpIOX::update_B_vi(){
  
  arma::mat Ytilde;
  B = arma::zeros(p, q);
  Beta_UQ = arma::zeros(p, q);
  
  if(latent_model > 0){
    Ytilde = (Y - W);
    for(int j=0; j<q; j++){
      // prior precision (diagonal)
      arma::vec vecB_Var = B_Var.col(j);
      arma::mat prior_prec = arma::diagmat(1.0 / vecB_Var);
      
      // posterior precision 
      arma::mat post_precision = prior_prec + X.t() * X;
      B_post_cov = arma::inv_sympd(post_precision);
      // posterior mean 
      arma::vec mu_b = B_post_cov * X.t() * Ytilde.col(j);
      
      // store the mean and optionally the covariance
      B.col(j) = mu_b;  
      Beta_UQ.col(j) = B_post_cov.diag();
      
    }
    
  } else {
    Ytilde = Y;
    Xtilde = arma::zeros(n*q, p*q);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; j++){
      Ytilde.col(j) = daggps.at(j).H_times_A( Ytilde.col(j) ); // whitened (Y - W)
      arma::mat HX = daggps.at(j).H_times_A(X);         // whitened X
      
      for(unsigned int i = 0; i < q; i++){
        Xtilde.submat(i * n, j * p,
                      (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX;
      }
    }
    
    arma::vec ytilde = arma::vectorise(Ytilde * Si); // vec(Y^* Σ^{-1})
    
    // prior precision (diagonal)
    arma::vec vecB_Var = arma::vectorise(B_Var);
    arma::mat prior_prec = arma::diagmat(1.0 / vecB_Var);
    
    // posterior precision 
    arma::mat post_precision = prior_prec + Xtilde.t() * Xtilde;
    B_post_cov = arma::inv_sympd(post_precision);
    // posterior mean 
    arma::vec mu_b = B_post_cov * Xtilde.t() * ytilde;
    
    // store the mean and optionally the covariance
    B = arma::mat(mu_b.memptr(), p, q);  // reshape to matrix for consistency
    arma::vec v = B_post_cov.diag();
    Beta_UQ = arma::mat(v.memptr(), p, q);
  }
  
  
  
  YXB = Y - X*B;
  
}
*/
bool SpIOX::upd_theta_metrop(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  theta_adapt.count_proposal();
  
  arma::vec phisig_cur = arma::vectorise( theta.rows(which_theta_elem) );
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(phisig_cur.n_elem);
  
  arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
    phisig_cur, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  arma::mat theta_alt = theta;
  arma::mat phisig_alt_mat = arma::mat(phisig_alt.memptr(), which_theta_elem.n_elem, q);
  
  theta_alt.rows(which_theta_elem) = phisig_alt_mat; 
  
  if(!theta_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  // ---------------------
  // create proposal daggp
  // this can run in parallel but update_theta already uses omp
  // do not run this in parallel, will be faster this way
  tstart = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<q; i++){
    daggps_alt[i].update_theta(theta_alt.col(i), true);
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "update dag proposal: " << timed << endl;
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
    arma::mat Target;
    if(latent_model>0){
      Target = W.col(j);
    } else {
      Target = YXB.col(j);
    }
    V_alt.col(j) = daggps_alt.at(j).H_times_A(Target);// * (Y.col(j) - X * B.col(j));
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
    daggp_alt_logdets(j) = daggps_alt.at(j).precision_logdeterminant;
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "computing V: " << timed << endl;
  
  tstart = std::chrono::steady_clock::now();
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
  //Rcpp::Rcout << "computing VSi: " << timed << endl;
  
  // priors
  double logpriors = 0;
  for(unsigned int j=0; j<q; j++){
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      // nned to change this prior if alpha is a proportion which we expect ~0
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
  }
  
  // ------------------
  // make move
  double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
  
  bool accepted = do_I_accept(logaccept);
  
  if(accepted){
    theta = theta_alt;
    std::swap(daggps, daggps_alt);
    std::swap(V, V_alt);
  } 
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
  }
  
  theta_mcmc_counter++;
  
  return accepted;
}

arma::uvec SpIOX::upd_theta_metrop_conditional(){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  arma::uvec accepteds = arma::zeros<arma::uvec>(q);
  
  for(int j=0; j<q; j++){
    c_theta_adapt[j].count_proposal();
    
    arma::vec phisig_cur = theta(which_theta_elem, oneuv*j);
    
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(phisig_cur.n_elem);
    
    arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
      phisig_cur, c_theta_unif_bounds) + 
        c_theta_adapt[j].paramsd * U_update, c_theta_unif_bounds);
    
    // proposal for theta matrix
    arma::mat theta_alt = theta;
    theta_alt(which_theta_elem, oneuv*j) = phisig_alt; 
    
    if(!theta_alt.is_finite()){
      Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
    }
    
    // ---------------------
    // create proposal daggp
    daggps_alt[j].update_theta(theta_alt.col(j), true);
    // conditional density of Y_j | Y_-j (or W depending on target)
    arma::mat V_alt = V;
    if(latent_model>0){
      V_alt.col(j) = daggps_alt.at(j).H_times_A(W.col(j));// * (Y.col(j) - X * B.col(j));
    } else {
      V_alt.col(j) = daggps_alt.at(j).H_times_A(YXB.col(j));// * (Y.col(j) - X * B.col(j));
    }
    
    double c_daggp_logdet = daggps.at(j).precision_logdeterminant;
    double c_daggp_alt_logdet = daggps_alt.at(j).precision_logdeterminant;
    
    arma::vec Vjc = arma::zeros(n);
    arma::vec Vjc_alt = arma::zeros(n);
    for(int jc=0; jc<q; jc++){
      Vjc += Q(j, jc)/Q(j,j) * V.col(jc);
      Vjc_alt += Q(j, jc)/Q(j,j) * V_alt.col(jc);
    }
    double core_alt = arma::accu(pow(Vjc_alt, 2.0)); 
    double core = arma::accu(pow(Vjc, 2.0)); 
    double prop_logdens = 0.5 * c_daggp_alt_logdet - Q(j,j)/2.0 * core_alt;
    double curr_logdens = 0.5 * c_daggp_logdet - Q(j,j)/2.0 * core;
    
    // priors
    double logpriors = 0;
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt, phisig_cur, c_theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    accepteds(j) = do_I_accept(logaccept);
    
    if(accepteds(j)){
      theta = theta_alt;
      std::swap(daggps.at(j), daggps_alt.at(j));
      //std::swap(V, V_alt);
      V.col(j) = V_alt.col(j);
    } 
    
    c_theta_adapt[j].update_ratios();
    
    if(theta_adapt_active){
      c_theta_adapt[j].adapt(U_update, exp(logaccept), theta_mcmc_counter); 
    }
    
    theta_mcmc_counter++;
  }
  
  return accepteds;
}

// wall clock in seconds
static inline double wall_time() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
#endif
}

void SpIOX::cache_blanket_comps(const arma::uvec& theta_changed){
  // all these matrix operations only need to be performed when theta changes
  // otherwise, we can just cache what's needed and reuse
  // this impacts the latent model, single site sampler
  // and imputation of missing data in the response model.
  int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;
  
  // allocate timing buffers BEFORE the loop
  arma::vec t_total(nfill, arma::fill::zeros);
  arma::vec t_rw(nfill,    arma::fill::zeros);
  arma::vec t_hit(nfill,   arma::fill::zeros);
  arma::vec t_pblk(nfill,  arma::fill::zeros);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) //***
#endif
  for(int ix=0; ix<nfill; ix++){
    const double t_start = wall_time();
    int i = latent_model == 2 ? ix : rows_with_missing(ix);
    
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
  
    // initialization
    Rw_no_Q(ix) = arma::zeros(q, q); 
    Pblanket_no_Q(ix) = arma::zeros(q, q*mbsize);

    // --- time Rw_no_Q ---
    double t0 = wall_time();
    for(int r=0; r<q; r++){
      for(int s=0; s<=r; s++){
        Rw_no_Q(ix)(r, s) = 
          arma::accu( daggps[r].H.col(i) % 
          daggps[s].H.col(i) );
        if(s!=r){
          Rw_no_Q(ix)(s, r) = Rw_no_Q(ix)(r, s);
        }
      }
    }
    double t1 = wall_time();
    t_rw(ix) = t1 - t0;
  
    // --- time building H_i_mat ---
    t0 = wall_time();
    arma::sp_mat H_i_mat(n, q);
    for (int r = 0; r < q; ++r) {
      H_i_mat.col(r) = daggps[r].H.col(i);
    }
    arma::sp_mat Hitt = H_i_mat.t();
    t1 = wall_time();
    t_hit(ix) = t1 - t0;
    // --- time Pblanket_no_Q block products ---
    //Rcpp::Rcout << arma::size(H_i_mat) << " " << arma::size(Pblanket_no_Q(ix)) << endl;
    t0 = wall_time();
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      arma::sp_mat Hblanket = daggps[s].H.cols(mblanket);
      arma::mat result(Hitt * Hblanket);
      Pblanket_no_Q(ix).cols(startcol, endcol) = result;
    }
    t1 = wall_time();
    t_pblk(ix) = t1 - t0;
  
  t_total(ix) = wall_time() - t_start;
  }
  // OPTIONAL: print after the loop (single-threaded) to avoid interleaving
  // for (int ix = 0; ix < nfill; ++ix) {
  //   Rcpp::Rcout << " total=" << arma::accu(t_total)
  //               << " Rw="    << arma::accu(t_rw)
   //              << " H_i="   << arma::accu(t_hit)
   //              << " Pblk="  << arma::accu(t_pblk) << "\n";
}


void SpIOX::w_sequential_singlesite(const arma::uvec& theta_changed){
  double ms_if_cache = 0;
  double ms_omp_for = 0;
  double ms_sample = 0;
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  arma::field<arma::mat> invcholP(n);
  
  // V = whitened Y-XB or W
  {
    auto t0 = std::chrono::steady_clock::now();
    if(arma::any(theta_changed != 0)){
      // perform this update if theta has changed and we need to recompute the
      // GP-related matrices that depend on it
      cache_blanket_comps(theta_changed);
    }
    ms_if_cache += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  
  {
    auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int i=0; i<n; i++){
      Rw(i) = Q % Rw_no_Q(i);
      
      // assume all the same dag otherwise we go cray
      arma::uvec mblanket = daggps[0].mblanket(i);
      int mbsize = mblanket.n_elem;
      arma::mat Pblanket = arma::zeros(q, q*mbsize);
      
      arma::mat Di_obs = arma::zeros(q,q);
      
      for(int j = 0; j < q; j++) {
        int startcol = j * mbsize;
        int endcol = (j + 1) * mbsize - 1;
        Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(j)) * Pblanket_no_Q(i).cols(startcol, endcol);
        if(!missing_mat(i,j)){
          Di_obs(j,j) = 1.0/Dvec(j);
        }
      }
      
      Hw(i) = - Pblanket;
      invcholP(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di_obs, "lower")));
    }
    
    ms_omp_for += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  
  {
    auto t0 = std::chrono::steady_clock::now();
    // visit every location and sample from latent effects 
    // conditional on data and markov blanket
  
    arma::mat mvnorm = arma::randn(q, n);
    for(int c=0; c < daggps[0].colors.n_elem; c++){
      arma::uvec nodes_in_color = daggps[0].colors(c);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for(int ix=0; ix < nodes_in_color.n_elem; ix++){
        int i = nodes_in_color(ix);
        //for(int i=0; i<n; i++){
        // data contributions may be null if data missing
        arma::vec Di_YXB = arma::zeros(q);
        for(int j=0; j<q; j++){
          if(!missing_mat(i,j)){
            Di_YXB(j) = YXB(i,j)/Dvec(j);
          }
        }
        
        arma::uvec mblanket = daggps[0].mblanket(i);
      
        // sample
        arma::vec W_mean = invcholP(i).t()*invcholP(i) * ( Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di_YXB );
        W.row(i) = arma::trans(  W_mean + invcholP(i).t()*mvnorm.col(i) );
        
        if(W.has_nan()){
          Rcpp::stop("Found nan in W.\n");
        }
      }
    }
    ms_sample += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  
}

/*
void SpIOX::w_sequential_singlesite(const arma::uvec& theta_changed, bool vi=false){
  // stuff to be moved to SpIOX class for latent model
  arma::mat Di = arma::diagmat(1/Dvec);
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  arma::mat mvnorm = arma::randn(q, n);
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  arma::field<arma::mat> Ctchol(n);
  
  // V = whitened Y-XB or W
  
#ifdef _OPENMP
//#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    
    Rw(i) = arma::zeros(q, q); 
    arma::mat Pblanket(q, q*mbsize);
    for(int r=0; r<q; r++){
      for(int s=0; s<q; s++){
        Rw(i)(r, s) = Q(r,s) * 
          arma::accu( daggps[r].H.col(i) % 
          daggps[s].H.col(i) );
        
        int startcol = s * mbsize;
        int endcol = (s + 1) * mbsize - 1;
        for(int j = 0; j < mblanket.n_elem; j++) {
          int col_idx = mblanket(j);
          Pblanket(r, startcol + j) = Q(r, s) *
            arma::accu(daggps[r].H.col(i) %
            daggps[s].H.col(col_idx));
        }
      }
    }
    
    Hw(i) = - Pblanket;
    Ctchol(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di, "lower")));
  }
  
  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int i=0; i<n; i++){
    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::vec meancomp = Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di * arma::trans(YXB.row(i)); 
    
    W.row(i) = arma::trans( Ctchol(i).t() * (Ctchol(i) * meancomp + 0*mvnorm.col(i) ));
  }
  
}
*/

void SpIOX::gibbs_w_sequential_byoutcome(){
  
  std::vector<arma::sp_mat> prior_precs(q);
  //std::vector<arma::spsolve_factoriser> factoriser(q);
  arma::uvec statuses = arma::ones<arma::uvec>(q);
  
  //arma::superlu_opts opts;
  //opts.symmetric  = true;
  
  arma::mat urands = arma::randn(n, q);
  arma::mat vrands = arma::randn(n, q);
  
  arma::mat HDs = arma::zeros(n, q);
  arma::uvec r1q = arma::regspace<arma::uvec>(0,q-1);
  
  for(int j=0; j<q; j++){
    // compute prior precision
    arma::sp_mat post_prec = Q(j,j) * daggps[j].Ci; //daggps[j].H.t() * daggps[j].H;
    // compute posterior precision
    post_prec.diag() += 1.0/Dvec(j);
    
    // this does not need sequential
    HDs.col(j) = YXB.col(j)/Dvec(j) + 
      urands.col(j)/sqrt(Dvec(j)) + 
      sqrt(Q(j,j)) * daggps[j].H.t() * vrands.col(j);
  
    arma::vec x = arma::zeros(n);
    arma::uvec notj = arma::find(r1q != j);
    arma::uvec jx = arma::zeros<arma::uvec>(1) + j;
    // V is whitened W as we want
    arma::vec Mi_m_prior = - daggps[j].H.t() * V.cols(notj) * Q.submat(notj, jx);
    arma::vec rhs = Mi_m_prior + HDs.col(j);
    
    arma::vec xguess = W.col(j);
    arma::vec w_sampled = gauss_seidel_solve(post_prec, rhs, xguess, 1e-3, 100);
    
    W.col(j) = w_sampled - arma::mean(w_sampled);
    V.col(j) = daggps.at(j).H_times_A(W.col(j));
  }
  
}

void SpIOX::gibbs_w_block(){
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
      daggps[i].H.t();
    row_offset += n;
    
    Unorm.col(i) = daggps[i].H.t() * Unorm.col(i);
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
  
  arma::vec wbefore = arma::vectorise(W);
  arma::vec w = gauss_seidel_solve(post_prec, post_meansample, wbefore, 1e-3, 100);

  W = arma::mat(w.memptr(), n, q);
  
}

void SpIOX::update_Dvec_gibbs(){
  arma::mat E = YXB - W;
  
  // priors fpr tau_sq
  double a = 2;//1e-5;
  double b = 1;//1e-5;
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    arma::mat Xj = X.rows(ix);
    arma::mat XtX = Xj.t() * Xj;
    
    double navail = .0 + avail_by_outcome(j).n_elem;

    // MCMC - sample with inverse gamma for each tau_sq
    arma::vec ej = E.col(j);
    
    double ssq = arma::accu(arma::square(ej.rows(ix)));
    Dvec(j) = 1.0/R::rgamma(navail/2 + a, 1.0/(b + 0.5 * ssq));
  
  }
}

void SpIOX::update_Dvec_vi(){
  
  // priors fpr tau_sq
  double a = 2;//1e-5;
  double b = 1;//1e-5;
  
  Dvec_UQ = arma::zeros(q);
  
  ETE = arma::zeros(q, q);
  //for(int ss=0; ss<N_mcvi_samples; ss++){
    arma::mat E = YXB - W; //W_samples_vi.slice(ss);
    for(int j=0; j<q; j++){
      E.col(j) -= arma::mean(E.col(j));
    }
    ETE = E.t() * E; //1.0/N_mcvi_samples * E.t() * E;
  //}
  
  update_running_means(ETE_ma, ETE);
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    arma::mat Xj = X.rows(ix);
    arma::mat XtX = Xj.t() * Xj;
    
    Dvec_UQ(j) = 0;//arma::accu(arma::square(ej.rows(ix)));
    double navail = .0 + avail_by_outcome(j).n_elem;
  
    // add sum_i Var(W_{i,j}) for the uncertainty of W
    Dvec_UQ(j) += ETE_ma(j,j);

    // posterior precision 
    
    //arma::mat post_precision = arma::diagmat(1.0 / B_Var.col(j)) + X.t() * X;
    //arma::mat Sbj = arma::inv_sympd(post_precision);
    //Dvec_UQ(j) += arma::trace(Sbj * XtX);
    
    // VI - posterior mean update for each tau_sq
    Dvec(j) = (b + 0.5 * Dvec_UQ(j)) / (navail/2 + a - 1);

  }
}

void SpIOX::W_centering(){
  if(intercept != -1){
    // we have an intercept. move the mean of W to it
    arma::rowvec w_means = arma::mean(W, 0);
    W.each_row() -= w_means;
    B.row(intercept) += w_means;
    YXB.each_row() -= w_means;
  }
}

void SpIOX::update_Sigma_iwishart(){
  arma::mat Smean = V.t() * V + arma::eye(V.n_cols, V.n_cols);
  arma::mat Q_mean_post;
  try { 
    Q_mean_post = arma::inv_sympd(Smean);
  } catch (...) {
    Rcpp::Rcout << Smean << endl;
    Rcpp::Rcout << theta << endl;
    Rcpp::stop("Error in inv_sympd within sample_Sigma_iwishart \n");
  }
   
  double df_post = n + (V.n_cols);
  
  Q = arma::wishrnd(Q_mean_post, df_post);
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  Sigma = S.t() * S;
}

void SpIOX::update_Sigma_vi(){

  if(latent_model > 0){
    Sigma_UQ = arma::eye(q,q) + VTV_ma;
    
  } else {
    // trace terms
    arma::mat E2 = arma::zeros(q, q);
    for (unsigned int i = 0; i < q; ++i){
      arma::mat Xi = Xtilde.rows(i * n, (i+1) * n - 1);
      for (unsigned int j = 0; j < q; ++j){
        arma::mat Xj = Xtilde.rows(j * n, (j+1) * n - 1);
        E2(i,j) = arma::trace(Xi * Beta_UQ * Xj.t());
      }
    }
    Sigma_UQ = arma::eye(q,q) + V.t()*V + E2;
  }
  
  double df_post = q + n;
  Sigma = Sigma_UQ / (df_post - q - 1);  
  Q = arma::inv_sympd(Sigma);
  Si = arma::chol(Q, "lower");
}


void SpIOX::sample_Y_misaligned(const arma::uvec& theta_changed){
  // precompute stuff in parallel so we can do fast sequential sampling after
  int nfill = rows_with_missing.n_elem;
  arma::mat mvnorm = arma::randn(q, nfill);
  
  arma::field<arma::mat> Hw(nfill);
  arma::field<arma::mat> Rw(nfill);
  arma::field<arma::mat> invcholP(nfill);
  
  
  if(arma::any(theta_changed != 0)){
    // perform this update if theta has changed and we need to recompute the
    // GP-related matrices that depend on it
    cache_blanket_comps(theta_changed);
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int ix=0; ix<nfill; ix++){
    // this row has missing data
    int i = rows_with_missing(ix);
    Rw(ix) = Q % Rw_no_Q(ix);
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(s)) * Pblanket_no_Q(ix).cols(startcol, endcol);
    }
    Hw(ix) = - Pblanket;
    invcholP(ix) = arma::inv(arma::trimatl(arma::chol(Rw(ix), "lower")));
  }
  
  // visit every location with missing data and fill 
  // conditional on what's available and markov blanket
  for(int ix=0; ix<nfill; ix++){
    arma::vec rnormq = mvnorm.col(ix); // preloaded sample size q
    // this row has missing
    int i = rows_with_missing(ix);
    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::mat YXB_others = YXB.rows(mblanket);
    
    arma::uvec which_missing = arma::find(missing_mat.row(i) == 1);
    if(which_missing.n_elem == q){
      // everything missing!
      arma::mat meancomp = invcholP(ix) * Hw(ix) * arma::vectorise( YXB_others );
      Y.row(i) = arma::trans(invcholP(ix).t() * (meancomp + rnormq)) + X.row(i) * B;
    } else {
      // some data available at this location
      arma::mat joint_cov = invcholP(ix).t() * invcholP(ix);
      arma::vec joint_mean = joint_cov * Hw(ix) * arma::vectorise( YXB_others );
      
      arma::uvec which_availab = arma::find(missing_mat.row(i) == 0);
      
      arma::mat Ckk = joint_cov(which_availab, which_availab);
      arma::mat Ckx = joint_cov(which_availab, which_missing);
      arma::mat Cxx = joint_cov(which_missing, which_missing);
      arma::mat HmatT = arma::solve(Ckk, Ckx);
      
      arma::mat cholRmat = arma::chol(arma::symmatu(Cxx - Ckx.t() * HmatT), "lower");
      
      arma::vec Yall = arma::trans(YXB.row(i));
      Yall(which_missing) = joint_mean(which_missing) + 
        HmatT.t() * (Yall(which_availab) - joint_mean(which_availab)) + 
          cholRmat * rnormq(which_missing);
      
      Y.row(i) = Yall.t() + X.row(i) * B;
    }
  }
  
  YXB = Y - X*B;
}


void SpIOX::gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq){
  
  if(sample_sigma > 0){
    tstart = std::chrono::steady_clock::now();
    update_Sigma_iwishart();
    timings(2) += time_count(tstart); 
  }
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  arma::uvec theta_has_changed = arma::zeros<arma::uvec>(q);
  if(update_theta){
    if(q>2){
      theta_has_changed = upd_theta_metrop_conditional();
    } else {
      bool block_changed = upd_theta_metrop();
      theta_has_changed += block_changed;
    }
    if(arma::any(theta_has_changed != 0)){
      HX_needs_updating = true;
    }
  }
  timings(4) += time_count(tstart);
  
  
  if(sample_beta){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    update_B();
    timings(0) += time_count(tstart);  
    
    // need to recompute V only when V = Y-XB (response model)
    if(latent_model == 0){
      tstart = std::chrono::steady_clock::now();
      compute_V();
      timings(1) += time_count(tstart);
    }
  }
  if(latent_model > 0){
    tstart = std::chrono::steady_clock::now();
    if(latent_model == 1){
      gibbs_w_block();
      W_centering();
      compute_V();
    } 
    if(latent_model == 2){
      // redo_cache_blanket runs if update_theta=true
      w_sequential_singlesite(theta_has_changed); 
      W_centering();
      compute_V();
    }
    if(latent_model == 3){
      gibbs_w_sequential_byoutcome();
    }
    
    // ASIS
    update_BW_asis(B, W, true); // do sample
    YXB = Y - X*B;
    W_centering();
    compute_V();
    
    if(sample_tausq){
      update_Dvec_gibbs();
    }
    timings(5) += time_count(tstart);
  } else {
    tstart = std::chrono::steady_clock::now();
    // response model -- do we have missing data? if so, impute
    if(Y_needs_filling){
      // redo_cache_blanket runs if update_theta=true
      sample_Y_misaligned(theta_has_changed);
    }
    timings(5) += time_count(tstart);
  }
}


void SpIOX::response_vi(){
  update_B();
  // V = whitened Y-XB or W
  compute_V();
  update_Sigma_vi();
}

void SpIOX::latent_vi(){
  // mcmc(asis)-vi
  
  // sample B
  update_B();
  // sample W
  arma::uvec theta_changed = arma::zeros<arma::uvec>(q); 
  w_sequential_singlesite(theta_changed);
  // asis rotation on B, W
  update_BW_asis(B, W, true); // with sampling
  // center W and place on intercept, if applicable
  W_centering();
  // working V with sampled B and W
  compute_V();
  // Update E(Sigma) using MC approx of E(quad forms)
  update_Sigma_vi();
  // Update E(Dvec) using MC approx of E(quad forms)
  update_Dvec_vi(); 
  
  // compute covariance in B iteratively
  vi_Beta_UQ();
  
  // update running means for E(B) and E(W)
  update_running_means(E_B, B);
  update_running_means(E_W, W);
  
  vi_it ++;
}



void SpIOX::map(){
  arma::mat Ytilde = Y;
  
  Xtilde = arma::zeros(n*q, p*q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = daggps.at(j).H_times_A(Y.col(j)); // whitening
    arma::mat HX = daggps.at(j).H_times_A(X);         // whitened X
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX;
    }
  }
  
  arma::vec ytilde = arma::vectorise(Ytilde * Si); // vec(Y^* Σ^{-1})
  
  arma::vec vecB_Var = arma::vectorise(B_Var);
  arma::mat prior_prec = arma::diagmat(1.0 / vecB_Var);
  
  arma::mat post_precision = prior_prec + Xtilde.t() * Xtilde;
  arma::vec mu_b = arma::solve(post_precision, Xtilde.t() * ytilde);
  
  // store the mean and optionally the covariance (if needed for ELBO or sampling)
  B = arma::mat(mu_b.memptr(), p, q);  // reshape to matrix for consistency
  
  Beta_UQ = arma::inv_sympd(post_precision);
  
  // V = whitened Y-XB or W
  compute_V();
  
  // trace terms
  arma::mat E2 = arma::zeros(q, q);
  for (unsigned int i = 0; i < q; ++i){
    arma::mat Xi = Xtilde.rows(i * n, (i+1) * n - 1);
    for (unsigned int j = 0; j < q; ++j){
      arma::mat Xj = Xtilde.rows(j * n, (j+1) * n - 1);
      E2(i,j) = arma::trace(Xi * Beta_UQ * Xj.t());
    }
  }
  
  arma::mat S_post = arma::eye(q,q) + V.t()*V + E2;
  double df_post = q + n;
  
  Sigma = S_post / (df_post + q + 1); // mode of IW
  Q = arma::inv_sympd(Sigma);         
  Si = arma::chol(Q, "lower");       

}



double SpIOX::logdens_eval(){
  arma::vec vecYtilde = arma::vectorise(Y - X * B);
  arma::vec daggp_logdets = arma::zeros(q);
  for(unsigned int j=0; j<q; j++){
    arma::mat vYtlocal = vecYtilde.subvec(j*n, (j+1)*n-1);
    vecYtilde.subvec(j*n, (j+1)*n-1) = daggps.at(j).H_times_A(vYtlocal);
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
  }
  
  arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  arma::vec ytilde = arma::vectorise(Ytildemat * Si);
  
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(ytilde, 2.0));
  
  return curr_logdens;
}

