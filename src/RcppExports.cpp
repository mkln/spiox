// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Correlationc
arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, const arma::vec& theta, int covar, bool same);
RcppExport SEXP _spiox_Correlationc(SEXP coordsxSEXP, SEXP coordsySEXP, SEXP thetaSEXP, SEXP covarSEXP, SEXP sameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsx(coordsxSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsy(coordsySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type covar(covarSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    rcpp_result_gen = Rcpp::wrap(Correlationc(coordsx, coordsy, theta, covar, same));
    return rcpp_result_gen;
END_RCPP
}
// dl_update_variances
arma::vec dl_update_variances(const arma::vec& theta, double a, double b);
RcppExport SEXP _spiox_dl_update_variances(SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(dl_update_variances(theta, a, b));
    return rcpp_result_gen;
END_RCPP
}
// expcov
arma::mat expcov(const arma::mat& x, const arma::mat& y, double phi);
RcppExport SEXP _spiox_expcov(SEXP xSEXP, SEXP ySEXP, SEXP phiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    rcpp_result_gen = Rcpp::wrap(expcov(x, y, phi));
    return rcpp_result_gen;
END_RCPP
}
// iox_svd
arma::mat iox_svd(const arma::mat& x, const arma::mat& y, int i, int j, const arma::mat& S, const arma::vec& philist, double cexp);
RcppExport SEXP _spiox_iox_svd(SEXP xSEXP, SEXP ySEXP, SEXP iSEXP, SEXP jSEXP, SEXP SSEXP, SEXP philistSEXP, SEXP cexpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type philist(philistSEXP);
    Rcpp::traits::input_parameter< double >::type cexp(cexpSEXP);
    rcpp_result_gen = Rcpp::wrap(iox_svd(x, y, i, j, S, philist, cexp));
    return rcpp_result_gen;
END_RCPP
}
// iox
arma::mat iox(const arma::mat& x, const arma::mat& y, int i, int j, const arma::mat& S, const arma::mat& theta, bool diag_only);
RcppExport SEXP _spiox_iox(SEXP xSEXP, SEXP ySEXP, SEXP iSEXP, SEXP jSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP diag_onlySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type diag_only(diag_onlySEXP);
    rcpp_result_gen = Rcpp::wrap(iox(x, y, i, j, S, theta, diag_only));
    return rcpp_result_gen;
END_RCPP
}
// iox_mat
arma::mat iox_mat(const arma::rowvec& x, const arma::rowvec& y, const arma::mat& S, const arma::vec& philist, double cexp);
RcppExport SEXP _spiox_iox_mat(SEXP xSEXP, SEXP ySEXP, SEXP SSEXP, SEXP philistSEXP, SEXP cexpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::rowvec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::rowvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type philist(philistSEXP);
    Rcpp::traits::input_parameter< double >::type cexp(cexpSEXP);
    rcpp_result_gen = Rcpp::wrap(iox_mat(x, y, S, philist, cexp));
    return rcpp_result_gen;
END_RCPP
}
// iox_mat_svd
arma::mat iox_mat_svd(const arma::rowvec& x, const arma::rowvec& y, const arma::mat& S, const arma::vec& philist, double cexp);
RcppExport SEXP _spiox_iox_mat_svd(SEXP xSEXP, SEXP ySEXP, SEXP SSEXP, SEXP philistSEXP, SEXP cexpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::rowvec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::rowvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type philist(philistSEXP);
    Rcpp::traits::input_parameter< double >::type cexp(cexpSEXP);
    rcpp_result_gen = Rcpp::wrap(iox_mat_svd(x, y, S, philist, cexp));
    return rcpp_result_gen;
END_RCPP
}
// iox_precomp
arma::mat iox_precomp(const arma::mat& x, const arma::mat& y, int i, int j, const arma::field<arma::mat>& Li_invs, const arma::mat& S, const arma::mat& theta);
RcppExport SEXP _spiox_iox_precomp(SEXP xSEXP, SEXP ySEXP, SEXP iSEXP, SEXP jSEXP, SEXP Li_invsSEXP, SEXP SSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type Li_invs(Li_invsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(iox_precomp(x, y, i, j, Li_invs, S, theta));
    return rcpp_result_gen;
END_RCPP
}
// iox_cross_avg
arma::vec iox_cross_avg(const arma::vec& hlist, int var_i, int var_j, const arma::mat& test_coords, const arma::mat& S, const arma::mat& theta, int num_angles, int num_threads);
RcppExport SEXP _spiox_iox_cross_avg(SEXP hlistSEXP, SEXP var_iSEXP, SEXP var_jSEXP, SEXP test_coordsSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP num_anglesSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type hlist(hlistSEXP);
    Rcpp::traits::input_parameter< int >::type var_i(var_iSEXP);
    Rcpp::traits::input_parameter< int >::type var_j(var_jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type test_coords(test_coordsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type num_angles(num_anglesSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(iox_cross_avg(hlist, var_i, var_j, test_coords, S, theta, num_angles, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// make_candidates
arma::umat make_candidates(const arma::mat& w, const arma::uvec& indsort, unsigned int col, double rho);
RcppExport SEXP _spiox_make_candidates(SEXP wSEXP, SEXP indsortSEXP, SEXP colSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type indsort(indsortSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type col(colSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(make_candidates(w, indsort, col, rho));
    return rcpp_result_gen;
END_RCPP
}
// neighbor_search_testset
arma::field<arma::uvec> neighbor_search_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho);
RcppExport SEXP _spiox_neighbor_search_testset(SEXP wtrainSEXP, SEXP wtestSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type wtrain(wtrainSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type wtest(wtestSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(neighbor_search_testset(wtrain, wtest, rho));
    return rcpp_result_gen;
END_RCPP
}
// radgp_build
Rcpp::List radgp_build(const arma::mat& coords, double rho, double phi, double sigmasq, double nu, double tausq, bool matern, int num_threads);
RcppExport SEXP _spiox_radgp_build(SEXP coordsSEXP, SEXP rhoSEXP, SEXP phiSEXP, SEXP sigmasqSEXP, SEXP nuSEXP, SEXP tausqSEXP, SEXP maternSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type tausq(tausqSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(radgp_build(coords, rho, phi, sigmasq, nu, tausq, matern, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// radgp_logdens
Rcpp::List radgp_logdens(const arma::vec& x, const arma::mat& coords, double rho, double phi, double sigmasq, double nu, double tausq, bool matern);
RcppExport SEXP _spiox_radgp_logdens(SEXP xSEXP, SEXP coordsSEXP, SEXP rhoSEXP, SEXP phiSEXP, SEXP sigmasqSEXP, SEXP nuSEXP, SEXP tausqSEXP, SEXP maternSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type tausq(tausqSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    rcpp_result_gen = Rcpp::wrap(radgp_logdens(x, coords, rho, phi, sigmasq, nu, tausq, matern));
    return rcpp_result_gen;
END_RCPP
}
// run_spf_model
Rcpp::List run_spf_model(arma::mat& Y, unsigned int n_factors, double delta_gamma_shape, double delta_gamma_rate, double dl_dirichlet_a, const arma::mat& Lambda_start, const arma::vec& Delta_start, unsigned int mcmc, int print_every, bool seq_lambda);
RcppExport SEXP _spiox_run_spf_model(SEXP YSEXP, SEXP n_factorsSEXP, SEXP delta_gamma_shapeSEXP, SEXP delta_gamma_rateSEXP, SEXP dl_dirichlet_aSEXP, SEXP Lambda_startSEXP, SEXP Delta_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP seq_lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type n_factors(n_factorsSEXP);
    Rcpp::traits::input_parameter< double >::type delta_gamma_shape(delta_gamma_shapeSEXP);
    Rcpp::traits::input_parameter< double >::type delta_gamma_rate(delta_gamma_rateSEXP);
    Rcpp::traits::input_parameter< double >::type dl_dirichlet_a(dl_dirichlet_aSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Lambda_start(Lambda_startSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Delta_start(Delta_startSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< bool >::type seq_lambda(seq_lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(run_spf_model(Y, n_factors, delta_gamma_shape, delta_gamma_rate, dl_dirichlet_a, Lambda_start, Delta_start, mcmc, print_every, seq_lambda));
    return rcpp_result_gen;
END_RCPP
}
// spiox_wishart
Rcpp::List spiox_wishart(const arma::mat& Y, const arma::mat& X, const arma::mat& coords, double radgp_rho, const arma::mat& theta_opts, const arma::mat& Sigma_start, const arma::mat& mvreg_B_start, int mcmc, int print_every, bool sample_iwish, bool sample_mvr, bool sample_gp, bool upd_opts, int num_threads);
RcppExport SEXP _spiox_spiox_wishart(SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP radgp_rhoSEXP, SEXP theta_optsSEXP, SEXP Sigma_startSEXP, SEXP mvreg_B_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP sample_iwishSEXP, SEXP sample_mvrSEXP, SEXP sample_gpSEXP, SEXP upd_optsSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type radgp_rho(radgp_rhoSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_opts(theta_optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma_start(Sigma_startSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mvreg_B_start(mvreg_B_startSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< bool >::type sample_iwish(sample_iwishSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_mvr(sample_mvrSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_gp(sample_gpSEXP);
    Rcpp::traits::input_parameter< bool >::type upd_opts(upd_optsSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_wishart(Y, X, coords, radgp_rho, theta_opts, Sigma_start, mvreg_B_start, mcmc, print_every, sample_iwish, sample_mvr, sample_gp, upd_opts, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_predict
Rcpp::List spiox_predict(const arma::mat& coords_new, const arma::mat& X_new, const arma::mat& Y, const arma::mat& X, const arma::mat& Xstar, const arma::mat& coords, double radgp_rho, const arma::mat& theta_options, const arma::cube& B, const arma::cube& S, const arma::umat& theta_which);
RcppExport SEXP _spiox_spiox_predict(SEXP coords_newSEXP, SEXP X_newSEXP, SEXP YSEXP, SEXP XSEXP, SEXP XstarSEXP, SEXP coordsSEXP, SEXP radgp_rhoSEXP, SEXP theta_optionsSEXP, SEXP BSEXP, SEXP SSEXP, SEXP theta_whichSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_new(coords_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_new(X_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Xstar(XstarSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type radgp_rho(radgp_rhoSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_options(theta_optionsSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type B(BSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type theta_which(theta_whichSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_predict(coords_new, X_new, Y, X, Xstar, coords, radgp_rho, theta_options, B, S, theta_which));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_spiox_Correlationc", (DL_FUNC) &_spiox_Correlationc, 5},
    {"_spiox_dl_update_variances", (DL_FUNC) &_spiox_dl_update_variances, 3},
    {"_spiox_expcov", (DL_FUNC) &_spiox_expcov, 3},
    {"_spiox_iox_svd", (DL_FUNC) &_spiox_iox_svd, 7},
    {"_spiox_iox", (DL_FUNC) &_spiox_iox, 7},
    {"_spiox_iox_mat", (DL_FUNC) &_spiox_iox_mat, 5},
    {"_spiox_iox_mat_svd", (DL_FUNC) &_spiox_iox_mat_svd, 5},
    {"_spiox_iox_precomp", (DL_FUNC) &_spiox_iox_precomp, 7},
    {"_spiox_iox_cross_avg", (DL_FUNC) &_spiox_iox_cross_avg, 8},
    {"_spiox_make_candidates", (DL_FUNC) &_spiox_make_candidates, 4},
    {"_spiox_neighbor_search_testset", (DL_FUNC) &_spiox_neighbor_search_testset, 3},
    {"_spiox_radgp_build", (DL_FUNC) &_spiox_radgp_build, 8},
    {"_spiox_radgp_logdens", (DL_FUNC) &_spiox_radgp_logdens, 8},
    {"_spiox_run_spf_model", (DL_FUNC) &_spiox_run_spf_model, 10},
    {"_spiox_spiox_wishart", (DL_FUNC) &_spiox_spiox_wishart, 14},
    {"_spiox_spiox_predict", (DL_FUNC) &_spiox_spiox_predict, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_spiox(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
