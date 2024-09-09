#include "omp_import.h"
#include "radgp.h"
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
  // Y = XB + Z where Z = WS is coregionalized multivariate q,q GP
  // S^T*S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1. Lambda, Q are sparse
  
  // -------------- data
  
  // matrix of outcomes dim n, q 
  arma::mat Y;
  // matrix of predictors dim n, p
  arma::mat X;
  
  // metadata
  int n, q, p;
  double spatial_sparsity;
  
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
  void sample_B(); // sample BSi 
  void compute_V(); // whitened
  void compute_S();
  void sample_theta();
  
  // -------------- run 1 gibbs iteration based on current values
  void gibbs_response(int sample_precision, bool sample_mvr, bool sample_gp);
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructors
  // for response model
  Inocs(const arma::mat& _Y, 
        const arma::mat& _X, 
        const arma::mat& _coords,
        
        double radgp_rho, const arma::mat& radgp_theta_options, 
        
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
    
    theta_options = radgp_theta_options;
    n_options = theta_options.n_cols;
    
    radgp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    for(unsigned int i=0; i<n_options; i++){
      radgp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1);
      //radgp_options.push_back( DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1) ); // 0=powerexp, 1 thread
    }
    spmap = arma::zeros<arma::uvec>(q); // outcome i uses radgp_options.at(spmap(i))
    spatial_sparsity = radgp_options.at(0).Ci.n_nonzero / (0.0 + n*n);
    
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
          
          double radgp_rho, const arma::mat& radgp_theta_options) 
  {
    model_response = true;
    
    Y = _Y;
    X = _X;
    
    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    theta_options = radgp_theta_options;
    n_options = theta_options.n_cols;
    
    radgp_options = std::vector<DagGP>(n_options);//.reserve(n_options);
    for(unsigned int i=0; i<n_options; i++){
      radgp_options[i] = DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1);
      //radgp_options.push_back( DagGP(_coords, theta_options.col(i), radgp_rho, 0, 1) ); // 0=powerexp, 1 thread
    }
  }  
  
};

inline void Inocs::compute_V(){
  // whiten the residuals from spatial dependence
  arma::mat residuals = ( Y - X * B ) * Si;
  V = arma::zeros(n, q);
  for(unsigned int i=0; i<q; i++){
    V.col(i) = radgp_options.at(spmap(i)).H * residuals.col(i);  
  }
  V = V * S; // restore original mv dependence
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
  
  //Rcpp::Rcout << "------- builds1 ----" << endl;
  tstart = std::chrono::steady_clock::now();
  arma::mat YS = Y * Si;
  // this samples BSi, then computes B by B = BSi * S;
  // this is a set of q independent univariate spatial regressions
  
  arma::mat rnormat = arma::randn(p, q);
  arma::mat BSi = arma::zeros(p, q);
  arma::mat Xty = arma::zeros(p, q);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<q; i++){
    arma::mat HX = radgp_options.at(spmap(i)).H * X;
    Xty.col(i) = X.t() * radgp_options.at(spmap(i)).Ci * YS.col(i);
    arma::mat Br_postp = arma::diagmat(1.0/B_Var.col(i)) + HX.t() * HX;
  
    arma::mat Br_postcholci = arma::inv(arma::trimatl(arma::chol(Br_postp, "lower")));
    BSi.col(i) = Br_postcholci.t() * (Br_postcholci * Xty.col(i) + rnormat.col(i));
  }
  
  B = BSi * S;

  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << timed << endl;
}

inline void Inocs::sample_theta(){
  // discrete uniform prior over spatial hyperparameters
  // all sparse matrices have been precomputed already

  arma::vec zz = arma::zeros(1);
  // whiten from multivariate dependence
  arma::mat Y_mv_white = ( Y - X * B ) * Si;
  // we now have independent columns with only spatial dependence
  arma::mat runif = arma::randu(q);
  for(unsigned int i=0; i<q; i++){
    // obviously, only do this for spatial variables
    arma::vec gp_logdens = arma::zeros(n_options);
    
    // calculate densities
    for(unsigned int j=0; j<n_options; j++){
      gp_logdens(j) = radgp_options.at(j).logdens(Y_mv_white.col(i));
    }
    //Rcpp::Rcout << i << " logdens: " << gp_logdens.t() << endl;
    // these probabilities are unnormalized
    double c = arma::max(gp_logdens);
    //Rcpp::Rcout << "max p: " << c << endl; 
    //Rcpp::Rcout << arma::trans(gp_logdens - c) << endl;
    double log_norm_const = c + log(arma::accu(exp(gp_logdens - c)));
  
    gp_logdens = exp(gp_logdens - log_norm_const);
    
    //Rcpp::Rcout << "norm logdens: " << gp_logdens.t() << endl;
    
    // finally sample
    double u = runif(i);
    arma::vec cprobs = arma::join_vert(zz, arma::cumsum(gp_logdens));
    // reassign process hyperparameters based on gp density
    spmap(i) = arma::max(arma::find(cprobs < u));
  }
}


inline void Inocs::gibbs_response(int sample_precision, bool sample_mvr, bool sample_gp){
  
  if(sample_mvr){
    //Rcpp::Rcout << "B " << endl;
    // sample B 
    tstart = std::chrono::steady_clock::now();
    sample_B();
    timings(0) += time_count(tstart);  
  }
  
  if(sample_precision > 0){
    //Rcpp::Rcout << "V " << endl;
    // compute V: spatial whitening of the random effects
    tstart = std::chrono::steady_clock::now();
    compute_V();
    timings(1) += time_count(tstart);;
    
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
  
  if(sample_gp){
    //Rcpp::Rcout << "T " << endl;
    // update theta | Y, S based on discrete prior
    tstart = std::chrono::steady_clock::now();
    sample_theta();
    timings(4) += time_count(tstart);
  }
  
}
