#include "spiox.h"

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
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
  if(tausq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = q * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 100}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({0.49, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
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
  } else {
    V = YXB;
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    V.col(j) = daggps.at(j).H_times_A(V.col(j));
  }
}

void SpIOX::sample_B(){
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
      Ytilde.col(j) = daggps.at(j).H_times_A(Y.col(j));// * Y.col(j);
      arma::mat HX = daggps.at(j).H_times_A(X);// * X;
      daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
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
    for(int j=0; j<q; j++){
      arma::vec yj = Ytilde.col(j);
      arma::vec y_available = yj.rows(avail_by_outcome(j));
      arma::vec XtYtildej = arma::trans(X.rows(avail_by_outcome(j))) * y_available;
      arma::mat post_precision = arma::diagmat(1.0/B_Var.col(j)) + XtX/Dvec(j);
      arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
      B.col(j) = pp_ichol.t() * (pp_ichol * XtYtildej/Dvec(j) + mvnorm.col(j));
    }
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::upd_theta_metrop(){
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
    if(tausq_sampling){
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
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
  
  //Rcpp::Rcout << theta.row(0) << endl;
  //Rcpp::Rcout << spmap.t() << endl;
}

void SpIOX::upd_theta_metrop_conditional(){
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
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
    if(tausq_sampling){
      logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt, phisig_cur, c_theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    bool accepted = do_I_accept(logaccept);
    
    if(accepted){
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
  
}

void SpIOX::gibbs_w_sequential_singlesite(){
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
    Rw(i) = arma::zeros(q, q); 
    for(int r=0; r<q; r++){
      for(int s=0; s<=r; s++){
        Rw(i)(r, s) = Q(r,s) * 
          arma::accu( daggps[r].H.col(i) % 
          daggps[s].H.col(i) );
        if(s!=r){
          Rw(i)(s, r) = Rw(i)(r, s);
        }
      }
    }
    
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    arma::sp_mat H_i_mat(n, q);
    for (int r = 0; r < q; ++r) {
      H_i_mat.col(r) = daggps[r].H.col(i);
    }
    
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(s)) * H_i_mat.t() * daggps[s].H.cols(mblanket);
    }

    Hw(i) = - Pblanket;
    arma::mat Di_obs = arma::zeros(q,q);
    for(int j=0; j<q; j++){
      if(!arma::is_finite(Y(i, j))){
        Di_obs(j,j) = 0;
      } else {
        Di_obs(j,j) = 1.0/Dvec(j);
      }
    }
    Ctchol(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di_obs, "lower")));
  }
  
  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int i=0; i<n; i++){
    
    // data contributions may be null if data missing
    arma::vec Di_YXB = arma::zeros(q);
    for(int j=0; j<q; j++){
      if(!arma::is_finite(Y(i, j))){
        Di_YXB(j) = 0;
      } else {
        Di_YXB(j) = YXB(i,j)/Dvec(j);
      }
    }

    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::vec meancomp = Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di_YXB;
    
    W.row(i) = arma::trans( Ctchol(i).t() * (Ctchol(i) * meancomp + mvnorm.col(i) ));
  }
  
  for(unsigned int i=0; i<q; i++){
    W.col(i) = W.col(i) - arma::mean(W.col(i));
  }
}

void SpIOX::gibbs_w_sequential_byoutcome(){
  
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
    arma::sp_mat post_prec = Q(j,j) * daggps[j].Ci; //daggps[j].H.t() * daggps[j].H;
    // compute posterior precision
    post_prec.diag() += 1.0/Dvec(j);
    
    // factorise precision so we can use it for sampling
    bool status = factoriser[j].factorise(post_prec, opts);
    if(status == false) { statuses(j) = 0; }
    
    // this does not need sequential
    HDs.col(j) = YXB.col(j)/Dvec(j) + 
      urands.col(j)/sqrt(Dvec(j)) + 
      sqrt(Q(j,j)) * daggps[j].H.t() * vrands.col(j);
  }
  
  if(arma::any(statuses==0)){ Rcpp::stop("Failed factorization for sampling w."); }
  
  arma::uvec r1q = arma::regspace<arma::uvec>(0,q-1);
  for(int j=0; j<q; j++){
    arma::vec x = arma::zeros(n);
    arma::uvec notj = arma::find(r1q != j);
    arma::uvec jx = arma::zeros<arma::uvec>(1) + j;
    // V is whitened W as we want
    arma::vec Mi_m_prior = - daggps[j].H.t() * V.cols(notj) * Q.submat(notj, jx);
    arma::vec rhs = Mi_m_prior + HDs.col(j);
    
    arma::vec w_sampled; 
    bool foundsol = factoriser[j].solve(w_sampled, rhs);
    
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

void SpIOX::sample_Dvec(){
  arma::mat E = YXB - W;
  for(int j=0; j<q; j++){
    arma::vec ej = E.col(j);
    double ssq = arma::accu(pow(ej.rows(avail_by_outcome(j)), 2));
    double navail = .0 + avail_by_outcome(j).n_elem;
    Dvec(j) = 1.0/R::rgamma(navail/2 + 1e-5, 1.0/(1e-5 + 0.5 * ssq));
  }
}

void SpIOX::sample_Sigma_iwishart(){
  arma::mat Smean = V.t() * V + arma::eye(V.n_cols, V.n_cols);
  arma::mat Q_mean_post = arma::inv_sympd(Smean);
  double df_post = n + (V.n_cols);
  
  Q = arma::wishrnd(Q_mean_post, df_post);
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  Sigma = S.t() * S;
}

void SpIOX::gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta){
  
  if(sample_sigma > 0){
    tstart = std::chrono::steady_clock::now();
    sample_Sigma_iwishart();
    timings(2) += time_count(tstart); 
  }
  
  if(sample_beta){
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
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  if(update_theta){
    if(q>2){
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


void SpIOX::vi(){
  arma::mat Ytilde = Y;
  arma::mat Xtilde = arma::zeros(n*q, p*q);
  
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
  
  // prior precision (diagonal)
  arma::vec vecB_Var = arma::vectorise(B_Var);
  arma::mat prior_prec = arma::diagmat(1.0 / vecB_Var);
  
  // posterior precision Λ_b = X̃ᵗ X̃ + prior precision
  arma::mat post_precision = prior_prec + Xtilde.t() * Xtilde;
  
  // posterior mean μ_b = Λ_b^{-1} X̃ᵗ ỹ
  arma::vec mu_b = arma::solve(post_precision, Xtilde.t() * ytilde);
  
  // store the mean and optionally the covariance (if needed for ELBO or sampling)
  B = arma::mat(mu_b.memptr(), p, q);  // reshape to matrix for consistency
  
  arma::mat B_post_cov = arma::inv_sympd(post_precision);
  
  // V = whitened Y-XB or W
  compute_V();
  
  // trace terms
  arma::mat E2 = arma::zeros(q, q);
  for (unsigned int i = 0; i < q; ++i){
    arma::mat Xi = Xtilde.rows(i * n, (i+1) * n - 1);
    for (unsigned int j = 0; j < q; ++j){
      arma::mat Xj = Xtilde.rows(j * n, (j+1) * n - 1);
      E2(i,j) = arma::trace(Xi * B_post_cov * Xj.t());
    }
  }
  
  arma::mat S_post = arma::eye(q,q) + V.t()*V + E2;
  double df_post = q + n;
  
  Sigma = S_post / (df_post - q - 1);         // Mean of IW
  Q = arma::inv_sympd(Sigma);         // Expectation of Σ⁻¹
  Si = arma::chol(Q, "lower");        // For B update


}


void SpIOX::map(){
  arma::mat Ytilde = Y;
  arma::mat Xtilde = arma::zeros(n*q, p*q);
  
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
  
  arma::mat B_post_cov = arma::inv_sympd(post_precision);
  
  // V = whitened Y-XB or W
  compute_V();
  
  // trace terms
  arma::mat E2 = arma::zeros(q, q);
  for (unsigned int i = 0; i < q; ++i){
    arma::mat Xi = Xtilde.rows(i * n, (i+1) * n - 1);
    for (unsigned int j = 0; j < q; ++j){
      arma::mat Xj = Xtilde.rows(j * n, (j+1) * n - 1);
      E2(i,j) = arma::trace(Xi * B_post_cov * Xj.t());
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

