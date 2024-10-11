#include "daggp.h"
#include <thread>

DagGP::DagGP(
  const arma::mat& coords_in,
  const arma::vec& theta_in,
  const arma::field<arma::uvec>& custom_dag,
  int covariance_model,
  int num_threads_in){
  
  coords = coords_in;
  theta = theta_in;
  nr = coords.n_rows;
  
  dag = custom_dag;
  
  oneuv = arma::ones<arma::uvec>(1);
  
  covar = covariance_model; // pexp or matern
  
  //thread safe stuff
  n_threads = num_threads_in;
  
  int bessel_ws_inc = MAT_NU_MAX;//see bessel_k.c for working space needs
  bessel_ws = (double *) R_alloc(n_threads*bessel_ws_inc, sizeof(double));
  
  //
  mblanket = arma::field<arma::uvec>(nr);
  hrows = arma::field<arma::vec>(nr);
  ax = arma::field<arma::uvec>(nr);
  compute_comps();
  initialize_H();
}


double DagGP::logdens(const arma::vec& x){
  double loggausscore = arma::conv_to<double>::from( x.t() * H.t() * H * x );
  return 0.5 * ( precision_logdeterminant - loggausscore );
}

void DagGP::update_theta(const arma::vec& newtheta, bool update_H){
  theta = newtheta;
  compute_comps(update_H);
}
void DagGP::compute_comps(bool update_H){
  // this function avoids building H since H is always used to multiply a matrix A
  sqrtR = arma::zeros(nr);
  h = arma::field<arma::vec>(nr);
  arma::vec logdetvec(nr);
  // calculate components for H and Ci
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    
    ax(i) = arma::join_vert(ix, px);
    arma::mat CC = Correlationf(coords, ix, ix, theta, bessel_ws, covar, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, bessel_ws, covar, false);
    arma::mat PPi = arma::inv_sympd( 
                Correlationf(coords, px, px, theta, bessel_ws, covar, true) );
    
    h(i) = PPi * CPt;
    sqrtR(i) = sqrt( arma::conv_to<double>::from(
      CC - CPt.t() * h(i) ));
    
    arma::vec mhR = -h(i)/sqrtR(i);
    arma::vec rowoneR = arma::ones(1)/sqrtR(i);
    
    hrows(i) = arma::join_vert(rowoneR, mhR);
    logdetvec(i) = -log(sqrtR(i));
    
    if(update_H){
      H(i,i) = 1.0/sqrtR(i);
      for(unsigned int j=0; j<dag(i).n_elem; j++){
        H(i, dag(i)(j)) = -h(i)(j)/sqrtR(i);
      }
    }
  }
  precision_logdeterminant = 2 * arma::accu(logdetvec);
}

void DagGP::initialize_H(){
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
      H_values(ix) =  -h(i)(j)/sqrtR(i);
      ix ++;
    }
  }
  H = arma::sp_mat(H_locs, H_values);
  
  // compute precision just to get the markov blanket
  arma::sp_mat Ci = H.t() * H;
  
  // comp markov blanket
  for(int i=0; i<nr; i++){
    int nonzeros_in_col = 0;
    for (arma::sp_mat::const_iterator it = Ci.begin_col(i); it != Ci.end_col(i); ++it) {
      nonzeros_in_col ++;
    }
    mblanket(i) = arma::zeros<arma::uvec>(nonzeros_in_col-1); // not the diagonal
    int ix = 0;
    for (arma::sp_mat::const_iterator it = Ci.begin_col(i); it != Ci.end_col(i); ++it) {
      if(it.row() != i){ 
        mblanket(i)(ix) = it.row();
        ix ++;
      }
    }
  }
}

arma::mat DagGP::H_times_A(const arma::mat& A, bool use_spmat){
  // this function avoids building H since H is always used to multiply a matrix A
  if(!use_spmat){
    arma::mat result = arma::zeros(nr, A.n_cols);
    // calculate components for H and Ci
    for(unsigned int j=0; j<A.n_cols; j++){
      arma::vec Aj = A.col(j);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
      for(int i=0; i<nr; i++){
        result(i, j) = arma::accu(hrows(i).t() * Aj(ax(i)));
      }
    }
    return result;
  } else {
    return H * A;
  }
}


arma::mat DagGP::Corr_export(const arma::mat& these_coords, const arma::uvec& ix, const arma::uvec& jx, bool same){
  return Correlationf(these_coords, ix, jx, theta, bessel_ws, covar, same);
}

//[[Rcpp::export]]
Rcpp::List daggp_build(const arma::mat& coords, const arma::field<arma::uvec>& dag,
                       double phi, double sigmasq, double nu, double tausq,
                       bool matern=false, int num_threads=1){
  
  arma::vec theta(4);
  theta(0) = phi;
  theta(1) = sigmasq;
  theta(2) = nu;
  theta(3) = tausq;
  
  int covar = matern;
  DagGP adag(coords, theta, dag, covar, num_threads);
  adag.compute_comps();
  adag.initialize_H();
  arma::sp_mat Ci = adag.H.t() * adag.H;
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("Cinv") = Ci,
    Rcpp::Named("Cinv_logdet") = adag.precision_logdeterminant
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