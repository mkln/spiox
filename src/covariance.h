
#ifndef XCOV_LMC 
#define XCOV_LMC

#include <RcppArmadillo.h>
#include "omp_import.h"

const int MAT_NU_MAX=3;

using namespace std;


arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, double * bessel_ws, bool matern, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, bool matern, bool same);

#endif
