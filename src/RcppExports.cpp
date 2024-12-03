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
arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, const arma::vec& theta, int matern, bool same);
RcppExport SEXP _spiox_Correlationc(SEXP coordsxSEXP, SEXP coordsySEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP sameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsx(coordsxSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsy(coordsySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    rcpp_result_gen = Rcpp::wrap(Correlationc(coordsx, coordsy, theta, matern, same));
    return rcpp_result_gen;
END_RCPP
}
// daggp_build
Rcpp::List daggp_build(const arma::mat& coords, const arma::field<arma::uvec>& dag, double phi, double sigmasq, double nu, double tausq, int matern, int num_threads);
RcppExport SEXP _spiox_daggp_build(SEXP coordsSEXP, SEXP dagSEXP, SEXP phiSEXP, SEXP sigmasqSEXP, SEXP nuSEXP, SEXP tausqSEXP, SEXP maternSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sigmasq(sigmasqSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type tausq(tausqSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(daggp_build(coords, dag, phi, sigmasq, nu, tausq, matern, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// iox
arma::mat iox(const arma::mat& x, const arma::mat& y, int i, int j, const arma::mat& S, const arma::mat& theta, bool matern, bool diag_only, bool at_limit);
RcppExport SEXP _spiox_iox(SEXP xSEXP, SEXP ySEXP, SEXP iSEXP, SEXP jSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP diag_onlySEXP, SEXP at_limitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< bool >::type diag_only(diag_onlySEXP);
    Rcpp::traits::input_parameter< bool >::type at_limit(at_limitSEXP);
    rcpp_result_gen = Rcpp::wrap(iox(x, y, i, j, S, theta, matern, diag_only, at_limit));
    return rcpp_result_gen;
END_RCPP
}
// sfact
arma::mat sfact(const arma::field<arma::uvec>& dag, const arma::mat& S, const arma::mat& theta, int matern, int n_threads);
RcppExport SEXP _spiox_sfact(SEXP dagSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(sfact(dag, S, theta, matern, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// make_ix
arma::uvec make_ix(int q, int n);
RcppExport SEXP _spiox_make_ix(SEXP qSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(make_ix(q, n));
    return rcpp_result_gen;
END_RCPP
}
// iox_mat
arma::mat iox_mat(const arma::mat& x, const arma::mat& y, const arma::mat& S, const arma::mat& theta, bool matern, bool D_only);
RcppExport SEXP _spiox_iox_mat(SEXP xSEXP, SEXP ySEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP D_onlySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< bool >::type D_only(D_onlySEXP);
    rcpp_result_gen = Rcpp::wrap(iox_mat(x, y, S, theta, matern, D_only));
    return rcpp_result_gen;
END_RCPP
}
// S_to_Sigma
arma::cube S_to_Sigma(const arma::cube& S);
RcppExport SEXP _spiox_S_to_Sigma(SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type S(SSEXP);
    rcpp_result_gen = Rcpp::wrap(S_to_Sigma(S));
    return rcpp_result_gen;
END_RCPP
}
// S_to_Q
arma::cube S_to_Q(const arma::cube& S);
RcppExport SEXP _spiox_S_to_Q(SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type S(SSEXP);
    rcpp_result_gen = Rcpp::wrap(S_to_Q(S));
    return rcpp_result_gen;
END_RCPP
}
// Sigma_to_correl
arma::cube Sigma_to_correl(const arma::cube& Sigma);
RcppExport SEXP _spiox_Sigma_to_correl(SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(Sigma_to_correl(Sigma));
    return rcpp_result_gen;
END_RCPP
}
// spiox_response
Rcpp::List spiox_response(const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& custom_dag, arma::mat theta_opts, const arma::mat& Sigma_start, const arma::mat& Beta_start, int mcmc, int print_every, int matern, bool sample_iwish, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts, int num_threads);
RcppExport SEXP _spiox_spiox_response(SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP custom_dagSEXP, SEXP theta_optsSEXP, SEXP Sigma_startSEXP, SEXP Beta_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP maternSEXP, SEXP sample_iwishSEXP, SEXP sample_mvrSEXP, SEXP sample_theta_gibbsSEXP, SEXP upd_theta_optsSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type custom_dag(custom_dagSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_opts(theta_optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma_start(Sigma_startSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Beta_start(Beta_startSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_iwish(sample_iwishSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_mvr(sample_mvrSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_theta_gibbs(sample_theta_gibbsSEXP);
    Rcpp::traits::input_parameter< bool >::type upd_theta_opts(upd_theta_optsSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_response(Y, X, coords, custom_dag, theta_opts, Sigma_start, Beta_start, mcmc, print_every, matern, sample_iwish, sample_mvr, sample_theta_gibbs, upd_theta_opts, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_latent
Rcpp::List spiox_latent(const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& custom_dag, arma::mat theta_opts, const arma::mat& Sigma_start, const arma::mat& Beta_start, int mcmc, int print_every, int matern, bool sample_iwish, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts, int num_threads, int sampling);
RcppExport SEXP _spiox_spiox_latent(SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP custom_dagSEXP, SEXP theta_optsSEXP, SEXP Sigma_startSEXP, SEXP Beta_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP maternSEXP, SEXP sample_iwishSEXP, SEXP sample_mvrSEXP, SEXP sample_theta_gibbsSEXP, SEXP upd_theta_optsSEXP, SEXP num_threadsSEXP, SEXP samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type custom_dag(custom_dagSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_opts(theta_optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma_start(Sigma_startSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Beta_start(Beta_startSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_iwish(sample_iwishSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_mvr(sample_mvrSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_theta_gibbs(sample_theta_gibbsSEXP);
    Rcpp::traits::input_parameter< bool >::type upd_theta_opts(upd_theta_optsSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< int >::type sampling(samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_latent(Y, X, coords, custom_dag, theta_opts, Sigma_start, Beta_start, mcmc, print_every, matern, sample_iwish, sample_mvr, sample_theta_gibbs, upd_theta_opts, num_threads, sampling));
    return rcpp_result_gen;
END_RCPP
}
// spiox_predict
Rcpp::List spiox_predict(const arma::mat& X_new, const arma::mat& coords_new, const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::cube& B, const arma::cube& Sigma, const arma::cube& theta, int matern, int num_threads);
RcppExport SEXP _spiox_spiox_predict(SEXP X_newSEXP, SEXP coords_newSEXP, SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP BSEXP, SEXP SigmaSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X_new(X_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_new(coords_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type B(BSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_predict(X_new, coords_new, Y, X, coords, dag, B, Sigma, theta, matern, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_predict_part
Rcpp::List spiox_predict_part(const arma::mat& Y_new, const arma::mat& X_new, const arma::mat& coords_new, const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::cube& B, const arma::cube& Sigma, const arma::cube& theta, int matern, int num_threads);
RcppExport SEXP _spiox_spiox_predict_part(SEXP Y_newSEXP, SEXP X_newSEXP, SEXP coords_newSEXP, SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP BSEXP, SEXP SigmaSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y_new(Y_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_new(X_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_new(coords_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type B(BSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_predict_part(Y_new, X_new, coords_new, Y, X, coords, dag, B, Sigma, theta, matern, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_latent_predict
Rcpp::List spiox_latent_predict(const arma::mat& X_new, const arma::mat& coords_new, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::cube& W, const arma::cube& B, const arma::cube& Sigma, const arma::mat& Dvec, const arma::cube& theta, int matern, int num_threads);
RcppExport SEXP _spiox_spiox_latent_predict(SEXP X_newSEXP, SEXP coords_newSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP WSEXP, SEXP BSEXP, SEXP SigmaSEXP, SEXP DvecSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X_new(X_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords_new(coords_newSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type W(WSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type B(BSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Dvec(DvecSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_latent_predict(X_new, coords_new, coords, dag, W, B, Sigma, Dvec, theta, matern, num_threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_spiox_Correlationc", (DL_FUNC) &_spiox_Correlationc, 5},
    {"_spiox_daggp_build", (DL_FUNC) &_spiox_daggp_build, 8},
    {"_spiox_iox", (DL_FUNC) &_spiox_iox, 9},
    {"_spiox_sfact", (DL_FUNC) &_spiox_sfact, 5},
    {"_spiox_make_ix", (DL_FUNC) &_spiox_make_ix, 2},
    {"_spiox_iox_mat", (DL_FUNC) &_spiox_iox_mat, 6},
    {"_spiox_S_to_Sigma", (DL_FUNC) &_spiox_S_to_Sigma, 1},
    {"_spiox_S_to_Q", (DL_FUNC) &_spiox_S_to_Q, 1},
    {"_spiox_Sigma_to_correl", (DL_FUNC) &_spiox_Sigma_to_correl, 1},
    {"_spiox_spiox_response", (DL_FUNC) &_spiox_spiox_response, 15},
    {"_spiox_spiox_latent", (DL_FUNC) &_spiox_spiox_latent, 16},
    {"_spiox_spiox_predict", (DL_FUNC) &_spiox_spiox_predict, 11},
    {"_spiox_spiox_predict_part", (DL_FUNC) &_spiox_spiox_predict_part, 12},
    {"_spiox_spiox_latent_predict", (DL_FUNC) &_spiox_spiox_latent_predict, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_spiox(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
