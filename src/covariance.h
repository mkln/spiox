
#ifndef XCOV_LMC 
#define XCOV_LMC

#include <RcppArmadillo.h>
#include "omp_import.h"

using namespace std;


arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, double * bessel_ws, int covar, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, double * bessel_ws, int covar, bool same);

#endif
