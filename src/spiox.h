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
  void sample_B(); // 
  void compute_V(); // whitened
  void compute_S();
  void sample_theta_discr(); // gibbs for each outcome choosing from options
  void upd_theta_metrop();
  void init_theta_adapt();
  
  // latent model 
  bool latent_model;
  arma::mat W;
  void gibbs_w_sequential();
  void gibbs_w_block();
  void sample_Dvec();
  arma::vec Dvec;
  arma::mat XtX;
  //
  
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
  void gibbs(int it, int sample_precision, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts);
  double logdens_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructors
  // for response model inverse Wishart
  SpIOX(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        bool response,
        const arma::mat& radgp_theta, 
        
        const arma::mat& Sigma_start,
        const arma::mat& mvreg_B_start,
        int num_threads_in) 
  {
    num_threads = num_threads_in;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    latent_model = !response; // latent model? 
    if(latent_model){
      W = arma::zeros(n, q);
      XtX = X.t() * X;
      Dvec = arma::ones(q)*.01;
    }
    
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
                               true, //matern 
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
  if(latent_model){
    V = W;
  } else {
    V = ( Y - X * B );
  }
  
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
  
  Sigma = S.t() * S;
  Q = Si * Si.t();
  // Sigma = S^T * S = (Si * Si^T)^-1
}

inline void SpIOX::sample_B(){
  
  if(!latent_model){
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
  
}

inline void SpIOX::sample_theta_discr(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  arma::vec zz = arma::zeros(1);
  
  arma::mat Target;
  if(latent_model){
    Target = W;
  } else {
    Target = Y - X * B;
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
  // create proposal radgp
  //Rcpp::Rcout << "------- builds1 ----" << endl;
  tstart = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int i=0; i<n_options; i++){
    daggp_options_alt[i].update_theta(theta_alt.col(i), true);
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
  
  // ----------------------
  // current density and proposal density
  //Rcpp::Rcout << "------- builds2 ----" << endl;
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
    arma::mat Target;
    if(latent_model){
      Target = W.col(j);
    } else {
      Target = Y.col(j) - X*B.col(j);
    }
    V_alt.col(j) = daggp_options_alt.at(spmap(j)).H_times_A(Target);// * (Y.col(j) - X * B.col(j));
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

inline void SpIOX::gibbs_w_sequential(){
  // stuff to be moved to SpIOX class for latent model
  arma::mat YXB = Y - X * B;
  arma::mat Di = arma::diagmat(1/Dvec);
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  arma::mat mvnorm = arma::randn(q, n);
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  arma::field<arma::mat> Ctchol(n);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggp_options[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    
    Rw(i) = arma::zeros(q, q); 
    
    arma::mat Pblank(q, q*mbsize);
    for(int r=0; r<q; r++){
      for(int s=0; s<q; s++){
        Rw(i)(r, s) = Q(r,s) * 
          arma::accu( daggp_options[spmap(r)].H.col(i) % 
          daggp_options[spmap(s)].H.col(i) );
        
        // Calculate the starting and ending column indices for Pblank
        int startcol = s * mbsize;
        int endcol = (s + 1) * mbsize - 1;
        
        // Fill the appropriate columns of Pblank
        Pblank.submat(r, startcol, r, endcol) = Q(r,s) * 
          arma::trans(daggp_options[spmap(r)].H.col(i)) *
          daggp_options[spmap(s)].H.cols(mblanket);
      }
    }
    
    Hw(i) = - //arma::inv_sympd(Rw(i)) * 
        Pblank;
  
    Ctchol(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di, "lower")));
  }

  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int i=0; i<n; i++){
    arma::uvec mblanket = daggp_options[0].mblanket(i);
    arma::vec meancomp = //Rw(i) * 
        Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di * arma::trans(YXB.row(i)); 
    
    W.row(i) = arma::trans( Ctchol(i).t() * (Ctchol(i) * meancomp + mvnorm.col(i) ));
  }
  
}

inline void SpIOX::gibbs_w_block(){
  
  // stuff to be moved to SpIOX class for latent model
  arma::mat YXB = Y - X * B;
  
  arma::sp_mat Hbig(q*n, q*n);
  
  // Offset for placing each matrix
  arma::uword row_offset = 0;
  arma::uword col_offset = 0;
  
  // Loop through each matrix and place it in the correct block
  for (int i=0; i<q; i++) {
    // Insert the matrix at the correct block position
    Hbig.submat(row_offset, col_offset, row_offset + n - 1, col_offset + n - 1) = 
      daggp_options[spmap(i)].H;
    
    // Update the offsets for the next block
    row_offset += n;
    col_offset += n;
  }
  
  arma::mat In = arma::eye(n,n);
  arma::mat Di = arma::diagmat(1/Dvec);
  arma::mat post_prec = Hbig.t() * arma::kron(Q, In) * Hbig + arma::kron(Di, In);
  arma::mat Ctchol = arma::inv(arma::trimatl(arma::chol(post_prec, "lower")));
  arma::vec post_Cmean = arma::kron(Di, In)*arma::vectorise(YXB);
  
  arma::vec w = Ctchol.t() * (Ctchol * post_Cmean + Ctchol * arma::randn(q*n));
  
  W = arma::mat(w.memptr(), n, q);
}


inline void SpIOX::sample_Dvec(){
  for(int i=0; i<q; i++){
    double ssq = arma::accu(pow( Y.col(i) - X*B.col(i) - W.col(i), 2));
    Dvec(i) = 1.0/R::rgamma(n/2 + 2, 1.0/(1 + 0.5 * ssq));
  }
}

inline void SpIOX::gibbs(int it, int sample_precision, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts){
  
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
      
      Q = arma::wishrnd(Q_mean_post, df_post);
      Si = arma::chol(Q, "lower");
      S = arma::inv(arma::trimatl(Si));
      Sigma = S.t() * S;
      
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
  
  if(latent_model){
    tstart = std::chrono::steady_clock::now();
    gibbs_w_sequential();
    sample_Dvec();
    compute_V();
    timings(6) += time_count(tstart);
  }
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

