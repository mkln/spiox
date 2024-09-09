#include "rinvgauss.h"

using namespace std;

arma::vec r_psi_cond(const arma::vec& phi, const arma::vec& theta, double tau);

double r_tau_cond(const arma::vec& phi, const arma::vec& theta, double a);

arma::vec r_phi_cond(const arma::vec& theta, double a);

arma::vec dl_update_variances(const arma::vec& theta, double a, double b=2);