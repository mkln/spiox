#include "omp_import.h"
#include "radgp.h"
#include "ramadapt.h"
#include "spf.h"
#include "cholesky_lrupd.h"

using namespace std;

inline int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

class Inocs {
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
  int n, q, p;
  //double spatial_sparsity;
  
  // -------------- model parameters
  arma::mat B;
  arma::mat B_Var; // prior variance on B, element by element
  double B_a_dl; // dirichlet-laplace parameter for vec(B)
  
  // SPF for sparse latent precision
  SparsePrecisionFactor spf;
  arma::mat S, Si; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // RadGP for spatial dependence
  std::vector<DagGP> radgp_q, radgp_q_alt;
  arma::mat theta; // one column per each outcome
  
  arma::mat V; 
  
  // -------------- utilities
  void compute_tildes();
  
  void sample_B(); // 
  void compute_V(); // whitened
  void compute_S();
  
  arma::mat Xtilde, Ytilde;
  arma::vec ytilde, radgp_logdets;
  
  arma::mat Ytilde_alt, Xtilde_alt;
  arma::vec radgp_alt_logdets;
  
  double current_logdens;
  void metrop_theta();
  
  // adaptive metropolis for theta
  int theta_mcmc_counter;
  arma::mat theta_unif_bounds;
  arma::mat theta_metrop_sd;
  RAMAdapt theta_adapt;
  bool theta_adapt_active;
  
  // -------------- run 1 gibbs iteration based on current values
  void gibbs_response(int sample_precision, bool sample_mvr, bool sample_gp);
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructors
  // for response model
  Inocs(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        
        double radgp_rho, const arma::vec& radgp_theta, 
        
        int spf_k, double spf_a_delta, double spf_b_delta, double spf_a_dl,
        
        const arma::mat& spf_Lambda_start, const arma::vec& spf_Delta_start,
        const arma::mat& mvreg_B_start) 
  {
    model_response = true;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    B = mvreg_B_start;
    B_Var = arma::ones(arma::size(B));
    
    theta = radgp_theta;
    theta_mcmc_counter = 0;
    theta_unif_bounds = arma::zeros(q, 2) + 0.1;
    theta_unif_bounds.col(1).fill(100);
    theta_metrop_sd = 0.05 * arma::eye(q,q);
    theta_adapt = RAMAdapt(theta.n_elem, theta_metrop_sd, 0.24);
    theta_adapt_active = true;
    
    Ytilde = Y;
    Xtilde = arma::zeros(n*q, p*q); // nq * pq
    radgp_logdets = arma::zeros(q);
    Ytilde_alt = Y;
    Xtilde_alt = Xtilde;
    radgp_alt_logdets = arma::zeros(q);
    
    radgp_q = std::vector<DagGP>(q);//.reserve(n_options);
    for(unsigned int i=0; i<q; i++){
      radgp_q[i] = DagGP(_coords, theta(i), radgp_rho, 0, 1);
      //radgp_q.push_back( DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1) ); // 0=powerexp, 1 thread
    }
    radgp_q_alt = radgp_q;
    //spatial_sparsity = radgp_q.at(0).Ci.n_nonzero / (0.0 + n*n);
    
    spf = SparsePrecisionFactor(&Y, spf_k, spf_a_delta, spf_b_delta, spf_a_dl);
    spf.Lambda_start(spf_Lambda_start);
    spf.Delta_start(spf_Delta_start);
    
    compute_S();
    
    timings = arma::zeros(10);
  }

  // for predictions
  Inocs(const arma::mat& _Y, 
          const arma::mat& _X, 
          const arma::mat& _coords,
          
          double radgp_rho, const arma::vec& radgp_theta) 
  {
    model_response = true;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    theta = radgp_theta;
    
    radgp_q = std::vector<DagGP>(q);//.reserve(n_options);
    for(unsigned int i=0; i<q; i++){
      radgp_q[i] = DagGP(_coords, theta(i), radgp_rho, 0, 1);
      //radgp_q.push_back( DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1) ); // 0=powerexp, 1 thread
    }
  }  
  
};

inline void Inocs::compute_V(){
  // whiten the residuals from spatial dependence
  V = ( Y - X * B );
  for(unsigned int i=0; i<q; i++){
    V.col(i) = radgp_q.at(i).H * V.col(i);  
  }
  // V = Li * (y-XB)
}

inline void Inocs::compute_S(){
  //arma::mat Q = spf.Lambda * spf.Lambda.t() + arma::diagmat(spf.Delta);
  //Si = arma::trimatl(arma::chol(Q, "lower"));
  
  // cholesky low rank update function
  arma::mat U = arma::diagmat(sqrt(spf.Delta));
  uchol_update(U, spf.Lambda);
  
  Si = U.t();
  S = arma::inv(arma::trimatl(Si));
  
  // Sigma = S^T * S = (Si * Si^T)^-1
}

inline void Inocs::compute_tildes(){
  radgp_logdets = arma::zeros(q);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = radgp_q.at(j).H * Y.col(j);
    arma::mat HX = radgp_q.at(j).H * X;
    radgp_logdets(j) = radgp_q.at(j).precision_logdeterminant;
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX; 
    }
  }
  ytilde = arma::vectorise(Ytilde * Si);
}

inline void Inocs::sample_B(){
  //Rcpp::Rcout << "+++++++++++++++++ ORIG +++++++++++++++++++" << endl;
  //S^T * S = Sigma
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  //Rcpp::Rcout << "------- builds0 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  
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
  
  double current_ldet = +0.5*arma::accu(radgp_logdets);
  current_logdens = current_ldet - 0.5*arma::accu(pow(ytilde - Xtilde * arma::vectorise(B), 2.0));
  
}

inline void Inocs::metrop_theta(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  arma::mat Sit = Si.t();
  
  theta_adapt.count_proposal();
  
  arma::vec theta_alt = theta;
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(theta_alt.n_elem);
  theta_alt = 
    par_huvtransf_back(
        par_huvtransf_fwd(theta, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  Rcpp::Rcout << "------- builds1 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  // create proposal radgp
  for(unsigned int i=0; i<q; i++){
    radgp_q_alt[i].update_theta(theta_alt(i));
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  Rcpp::Rcout << timed << endl;
  
  // proposal density
  Rcpp::Rcout << "------- builds2 ----" << endl;
  tstart = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde_alt.col(j) = radgp_q_alt.at(j).H * Y.col(j);
    arma::mat HX = radgp_q_alt.at(j).H * X;
    radgp_alt_logdets(j) = radgp_q_alt.at(j).precision_logdeterminant;
    for(unsigned int i=0; i<q; i++){
      Xtilde_alt.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Sit(i, j) * HX; 
    }
  }
  arma::vec ytilde_alt = arma::vectorise(Ytilde_alt * Si);
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  Rcpp::Rcout << timed << endl;
  
  Rcpp::Rcout << "------- builds3 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  double prop_ldet = +0.5*arma::accu(radgp_alt_logdets);
  double prop_logdens = prop_ldet 
    - 0.5*arma::accu(pow(ytilde_alt - Xtilde_alt * arma::vectorise(B), 2.0));
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  Rcpp::Rcout << timed << endl;
  
  double jacobian  = calc_jacobian(theta_alt, theta, theta_unif_bounds);
  
  double logaccept = prop_logdens - current_logdens + jacobian;
  bool accepted = do_I_accept(logaccept);
  
  
  if(accepted){
    theta = theta_alt;
    std::swap(radgp_q, radgp_q_alt);
  } 
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
  }
  
  theta_mcmc_counter++;
}


inline void Inocs::gibbs_response(int sample_precision, bool sample_mvr, bool sample_gp){
  
  if(sample_precision > 0){
    //Rcpp::Rcout << "V " << endl;
    // compute V: spatial whitening of the random effects
    tstart = std::chrono::steady_clock::now();
    compute_V();
    timings(1) += time_count(tstart);
    
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
      tstart = std::chrono::steady_clock::now();
      arma::mat S = n * arma::cov(V) + arma::eye(V.n_cols, V.n_cols);
      arma::mat Q_mean_post = arma::inv_sympd(S);
      double df_post = n + (V.n_cols);
      
      arma::mat Q = arma::wishrnd(Q_mean_post, df_post);
      
      Si = arma::chol(Q, "lower");
      S = arma::inv(arma::trimatl(Si));
      timings(2) += time_count(tstart); 
    }
  }
  
  if(sample_mvr | sample_gp){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    compute_tildes();
    timings(5) += time_count(tstart);  
  }
  
  
  if(sample_mvr){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    sample_B();
    timings(0) += time_count(tstart);  
  }
  
  if(sample_gp){
    //Rcpp::Rcout << "T " << endl;
    // update theta | Y, S based on discrete prior
    tstart = std::chrono::steady_clock::now();
    metrop_theta();
    timings(4) += time_count(tstart);
  }
  
}
