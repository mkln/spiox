#include "dirichletlaplace.h"

using namespace std;

arma::vec r_psi_cond(const arma::vec& phi, const arma::vec& theta, double tau){
  unsigned int p = phi.n_elem;
  arma::vec psi = arma::zeros(p);
  for(unsigned int i=0; i<p; i++){
    psi(i) = c_rigauss( phi(i) * tau / abs(theta(i)) , 1);
  }
  return psi;
}

double r_tau_cond(const arma::vec& phi, const arma::vec& theta, double a, double b){
  int p = theta.n_elem;
  double chi = 2*arma::accu(abs(theta)/phi);
  double lambda = p*a - p;
  double res = c_rgigauss_vec(b*2, lambda, chi, 1)(0);
  return res;
}

arma::vec r_phi_cond(const arma::vec& theta, double a){
  unsigned int p = theta.n_elem;
  arma::vec Tvec = arma::zeros(p);
  for(unsigned int i=0; i<p; i++){
    Tvec(i) = c_rgigauss_vec(1, a-1, 2 * abs(theta(i)), 1)(0);
  }
  double Tsum = arma::accu(Tvec);
  arma::vec phi = Tvec/Tsum;
  return phi;
}

//[[Rcpp::export]]
arma::vec dl_update_variances(const arma::vec& theta, double a, double b){
  // samples theta's variance given theta itself and the hyperparam a
  arma::vec phi = r_phi_cond(theta, a);
  double tau = r_tau_cond(phi, theta, a, b);
  arma::vec psi = r_psi_cond(phi, theta, tau);
    
    //Rcpp::Rcout << tau*tau << " " << arma::trans(psi%phi%phi) << endl;
  arma::vec theta_var = tau*tau * (psi%phi%phi);
  
  theta_var(arma::find(theta_var < 1e-8)).fill(1e-8);
  return theta_var;
}