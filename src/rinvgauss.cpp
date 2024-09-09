#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace std;


arma::vec c_rigauss_vec(int n, double mu, double lambda){
  arma::vec random_vector(n);
  double z,y,x,u;
  for(int i=0; i<n; ++i){
    z = R::rnorm(0,1);
    y = z*z;
    x = mu+0.5*mu*mu*y/lambda - 0.5 * (mu/lambda) * sqrt(4*mu*lambda*y+mu*mu*y*y);
    u = R::runif(0,1);
    if(u <= mu/(mu+x)){
      random_vector(i) = x;
    }else{
      random_vector(i) = mu*mu/x;
    };
  }
  return random_vector;
}

double c_rigauss(double mu, double lambda){
  double z = R::rnorm(0,1);
  double y = z*z;
  double x = mu+0.5*mu*mu*y/lambda - 0.5 * (mu/lambda) * sqrt(4*mu*lambda*y+mu*mu*y*y);
  double u = R::runif(0,1);
  double res;
  if(u <= mu/(mu+x)){
    res = x;
  } else {
    res = mu*mu/x;
  };
  return res;
}
