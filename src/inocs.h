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
  std::vector<DagGP> radgp_options;
  arma::mat theta_options; // each column is one alternative value for theta
  unsigned int n_options;
  arma::uvec spmap; // qx1 vector spmap(i) = which element of radgp_options for factor i
  
  arma::mat V; 
  
  // -------------- utilities
  void sample_B(); // 
  void compute_V(); // whitened
  void compute_S();
  void sample_theta(); // gibbs for each outcome choosing from options
  
  // -------------- run 1 gibbs iteration based on current values
  void gibbs_response(int sample_precision, bool sample_mvr, bool sample_gp);
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructors
  // for response model
  Inocs(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        
        double radgp_rho, const arma::mat& radgp_theta, 
        
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
    
    theta_options = radgp_theta;
    n_options = theta_options.n_cols;
    radgp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    for(unsigned int i=0; i<n_options; i++){
      radgp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1);
    }
    spmap = arma::zeros<arma::uvec>(q);
    
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
    radgp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    for(unsigned int i=0; i<n_options; i++){
      radgp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1);
    }
  }  
  
};

inline void Inocs::compute_V(){
  // whiten the residuals from spatial dependence
  V = ( Y - X * B );
  for(unsigned int i=0; i<q; i++){
    V.col(i) = radgp_options.at(spmap(i)).H * V.col(i);  
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

inline void Inocs::sample_B(){
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
#pragma omp parallel for 
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = radgp_options.at(spmap(j)).H * Y.col(j);
    arma::mat HX = radgp_options.at(spmap(j)).H * X;
    radgp_logdets(j) = radgp_options.at(spmap(j)).precision_logdeterminant;
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

inline void Inocs::sample_theta(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  arma::vec zz = arma::zeros(1);
  
  // initialize current setup -- should be able to start from values from B step
  arma::mat Ytilde = Y;
  arma::mat Xtilde = arma::zeros(n*q, p*q);
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = radgp_options.at(spmap(j)).H * Y.col(j);
    
    arma::mat HX = radgp_options.at(spmap(j)).H * X;
    //radgp_logdets(j) = radgp_options_alt.at(spmap(j)).precision_logdeterminant;
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j, i) * HX; 
    }
  }
  
  //Rcpp::Rcout << "theta opts " << endl << theta_options << endl;
  
  // loop over outcomes -- this is gibbs so cannot parallelize
  for(unsigned int j=0; j<q; j++){
    arma::vec opt_logdens = arma::zeros(n_options);
    // -- loop over options // parallel ok if we create a copy of ytilde and Xtilde
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int r=0; r<n_options; r++){
      // compute unnorm logdens 
      // change option for this outcome: spmap(i) is the new theta option
      
      //Rcpp::Rcout << "r: " << r << " spmap(r) " << spmap(r) << endl;
      //Rcpp::Rcout << "outer " << j*n << " " << (j+1) * n-1 << endl;
      
      //Rcpp::Rcout << "theta for this option " << radgp_options.at(r).theta.t() << endl;
      arma::mat Xtilde_loc = Xtilde;
      arma::mat Ytilde_loc = Ytilde;
      Ytilde_loc.col(j) = radgp_options.at(r).H * Y.col(j);
      arma::mat HX = radgp_options.at(r).H * X;
      double opt_prec_logdet = radgp_options.at(r).precision_logdeterminant;
      for(unsigned int i=0; i<q; i++){
        //Rcpp::Rcout << "inner " << j*p << " " << (j+1) * p-1 << endl;
        Xtilde_loc.submat(i * n,       j * p,
                      (i+1) * n-1, (j+1) * p - 1) = Si(j, i) * HX; 
      }
      arma::vec ytilde_loc = arma::vectorise(Ytilde_loc * Si);
      // at each i, we ytilde and Xtilde remain the same except for outcome j
      opt_logdens(r) = 0.5 * opt_prec_logdet 
                - 0.5*arma::accu(pow(ytilde_loc - Xtilde_loc * arma::vectorise(B), 2.0));
      
      //Rcpp::Rcout << "option accu: " << arma::accu(pow(ytilde - Xtilde * arma::vectorise(B), 2.0)) << endl;
    }
    
    // these probabilities are unnormalized
    //Rcpp::Rcout << "options probs unnorm: " << opt_logdens.t() << endl;
    
    double c = arma::max(opt_logdens);
    double log_norm_const = c + log(arma::accu(exp(opt_logdens - c)));
    // now normalize
    opt_logdens = exp(opt_logdens - log_norm_const);
    
    //Rcpp::Rcout << "options probs unnorm: " << opt_logdens.t() << endl;
    
    // finally sample
    double u = arma::randu();
    arma::vec cprobs = arma::join_vert(zz, arma::cumsum(opt_logdens));
    // reassign process hyperparameters based on gp density
    spmap(j) = arma::max(arma::find(cprobs < u));
    
    // update ytilde and Xtilde accordingly so we can move on to the next
    Ytilde.col(j) = radgp_options.at(spmap(j)).H * Y.col(j);
    arma::mat HX = radgp_options.at(spmap(j)).H * X;
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j, i) * HX; 
    }
    
    //Rcpp::Rcout << "done" << endl;
  }
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
    sample_theta();
    timings(4) += time_count(tstart);
  }
  
}
