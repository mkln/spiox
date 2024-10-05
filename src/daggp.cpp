#include "daggp.h"
#include <thread>

DagGP::DagGP(
    const arma::mat& coords_in, 
    const arma::vec& theta_in,
    double rho_in,
    int covariance_model,
    int num_threads_in){
  
  coords = coords_in;
  theta = theta_in;
  
  nr = coords.n_rows;
  rho = rho_in;
  
  layers = arma::zeros<arma::uvec>(nr);
  dag = radialndag(coords, rho);
  oneuv = arma::ones<arma::uvec>(1);
  
  covar = covariance_model; // pexp or matern

  //n_threads = nthread == 0? 1 : nthread;
  
  //thread safe stuff
  n_threads = num_threads_in;
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  bessel_ws = (double *) R_alloc(n_threads*bessel_ws_inc, sizeof(double));
  
  //initialize_H();
  compute_comps();
}


DagGP::DagGP(
  const arma::mat& coords_in,
  const arma::vec& theta_in,
  const arma::field<arma::uvec>& custom_dag,
  int covariance_model,
  int num_threads_in){
  
  coords = coords_in;
  theta = theta_in;
  
  nr = coords.n_rows;
  rho = -1;
  
  dag = custom_dag;
  oneuv = arma::ones<arma::uvec>(1);
  
  covar = covariance_model; // pexp or matern
  
  //n_threads = nthread == 0? 1 : nthread;
  
  //thread safe stuff
  n_threads = num_threads_in;
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  bessel_ws = (double *) R_alloc(n_threads*bessel_ws_inc, sizeof(double));
  
  //initialize_H();
  compute_comps();
}



double DagGP::logdens(const arma::vec& x){
  double loggausscore = arma::conv_to<double>::from( x.t() * H.t() * H * x );
  return 0.5 * ( precision_logdeterminant - loggausscore );
}

void DagGP::update_theta(const arma::vec& newtheta){
  theta = newtheta;
  //initialize_H();
  compute_comps();
}

void DagGP::initialize_H(){
  arma::field<arma::vec> ht(nr);
  arma::vec sqrtR(nr);
  arma::vec logdetvec(nr);
  
  // calculate components for H and Ci
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, bessel_ws, covar, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, covar, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, bessel_ws, covar, true) );
    
    ht(i) = PPi * CPt;
    sqrtR(i) = sqrt( arma::conv_to<double>::from(
                                          CC - CPt.t() * ht(i) ));
    logdetvec(i) = -log(sqrtR(i));
  }

  // building sparse H and Ci
  unsigned int H_n_elem = nr;
  for(int i=0; i<nr; i++){
    H_n_elem += dag(i).n_elem;
  }
  arma::umat H_locs(2, H_n_elem);
  arma::vec H_values(H_n_elem);
  unsigned int ix=0;
  for(int i=0; i<nr; i++){
    H_locs(0, ix) = i;
    H_locs(1, ix) = i;
    H_values(ix) = 1.0/sqrtR(i);
    ix ++;
    for(unsigned int j=0; j<dag(i).n_elem; j++){
      H_locs(0, ix) = i;
      H_locs(1, ix) = dag(i)(j);
      H_values(ix) =  -ht(i)(j)/sqrtR(i);
      ix ++;
    }
  }

  precision_logdeterminant = 2 * arma::accu(logdetvec);
  H = arma::sp_mat(H_locs, H_values);
  //Ci = H.t() * H;
}

void DagGP::compute_comps(){
  // this function avoids building H since H is always used to multiply a matrix A
  
  arma::vec logdetvec(nr);
  hrows = arma::field<arma::rowvec>(nr);
  
  // calculate components for H and Ci
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    arma::uvec ax = arma::join_vert(ix, px);
    
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, bessel_ws, covar, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, covar, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, bessel_ws, covar, true) );
    
    arma::rowvec h = arma::trans(PPi * CPt);
    double sqrtR = sqrt( arma::conv_to<double>::from(
      CC - CPt.t() * h.t() ));
    
    arma::rowvec mhR = -h/sqrtR;
    arma::rowvec rowoneR = arma::ones<arma::rowvec>(1)/sqrtR;
    
    hrows(i) = arma::join_horiz(mhR, rowoneR);
    logdetvec(i) = -log(sqrtR);
  }
  precision_logdeterminant = 2 * arma::accu(logdetvec);
}

arma::mat DagGP::H_times_A(const arma::mat& A){
  // this function avoids building H since H is always used to multiply a matrix A
  
  arma::field<arma::vec> ht(nr);
  arma::vec sqrtR(nr);
  arma::vec logdetvec(nr);
  arma::mat result = arma::zeros(nr, A.n_cols);
  
  // calculate components for H and Ci
#ifdef _OPENMP
//#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    arma::uvec ax = arma::join_vert(px, ix);
    
    for(unsigned int j=0; j<A.n_cols; j++){
      result(i, j) = arma::conv_to<double>::from(hrows(i) * A.submat(ax, oneuv*j));
    }
  }
  return result;
}


arma::mat DagGP::Corr_export(const arma::mat& these_coords, const arma::uvec& ix, const arma::uvec& jx, bool same){
  return Correlationf(these_coords, ix, jx, theta, bessel_ws, covar, same);
}


//[[Rcpp::export]]
Rcpp::List radgp_build(const arma::mat& coords, double rho, 
                       double phi, double sigmasq, double nu, double tausq,
                       bool matern=false, int num_threads=1){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = tausq;
  
  int covar = matern;
  DagGP adag(coords, theta, rho, covar, num_threads);
  adag.initialize_H();
  arma::sp_mat Ci = adag.H.t() * adag.H;
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant,
    Rcpp::Named("layers") = adag.layers
  );
}

//[[Rcpp::export]]
Rcpp::List daggp_build_mm(const arma::mat& A, const arma::mat& coords, const arma::field<arma::uvec>& dag,
                       double phi, double sigmasq, double nu, double tausq,
                       bool matern=false, int num_threads=1){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = tausq;
  
  int covar = matern;
  DagGP adag(coords, theta, dag, covar, num_threads);
  adag.initialize_H();
  arma::sp_mat Ci = adag.H.t() * adag.H;
  
  adag.compute_comps();
  arma::vec result = adag.H_times_A(A);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant,
    Rcpp::Named("hrows") = adag.hrows,
    Rcpp::Named("result") = result
  );
}

//[[Rcpp::export]]
Rcpp::List radgp_logdens(const arma::vec& x, 
                       const arma::mat& coords, double rho, 
                       double phi, double sigmasq, double nu, double tausq,
                       bool matern=false){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = tausq;
  
  int covar = matern;
  DagGP adag(coords, theta, rho, covar);
  adag.initialize_H();
  arma::sp_mat Ci = adag.H.t() * adag.H;
  double logdens = adag.logdens(x);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant,
    Rcpp::Named("layers") = adag.layers,
    Rcpp::Named("logdens") = logdens
  );
}


/*

using EigenFromArma = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
 
arma::sp_mat sparse_convert(const Eigen::SparseMatrix<double>& XE){
 unsigned int num_cols = XE.cols();
 unsigned int num_rows = XE.rows();
 
 std::vector<unsigned int> rowind_vect(
 XE.innerIndexPtr(), XE.innerIndexPtr() + XE.nonZeros());
 std::vector<unsigned int> colptr_vect(
 XE.outerIndexPtr(), XE.outerIndexPtr() + XE.outerSize() + 1);
 std::vector<double> values_vect(
 XE.valuePtr(), XE.valuePtr() + XE.nonZeros());
 
 arma::vec values(values_vect.data(), values_vect.size(), false);
 arma::uvec rowind(rowind_vect.data(), rowind_vect.size(), false);
 arma::uvec colptr(colptr_vect.data(), colptr_vect.size(), false);
 
 arma::sp_mat result(rowind, colptr, values, num_rows, num_cols);
 
 return result; 
} */