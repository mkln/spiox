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
// daggp_build
Rcpp::List daggp_build(const arma::mat& coords, const arma::field<arma::uvec>& dag, double phi, double sigmasq, double nu, double tausq, bool matern, int num_threads);
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
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(daggp_build(coords, dag, phi, sigmasq, nu, tausq, matern, num_threads));
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
arma::mat sfact(const arma::field<arma::uvec>& dag, const arma::mat& S, const arma::mat& theta, bool matern, int n_threads);
RcppExport SEXP _spiox_sfact(SEXP dagSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP maternSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    Rcpp::traits::input_parameter< int >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(sfact(dag, S, theta, matern, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// rvec
arma::mat rvec(const arma::mat& x, int i, const arma::mat& S, const arma::mat& theta, bool matern);
RcppExport SEXP _spiox_rvec(SEXP xSEXP, SEXP iSEXP, SEXP SSEXP, SEXP thetaSEXP, SEXP maternSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type matern(maternSEXP);
    rcpp_result_gen = Rcpp::wrap(rvec(x, i, S, theta, matern));
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
Rcpp::List spiox_wishart(const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& custom_dag, arma::mat theta_opts, const arma::mat& Sigma_start, const arma::mat& mvreg_B_start, int mcmc, int print_every, bool sample_iwish, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts, int num_threads);
RcppExport SEXP _spiox_spiox_wishart(SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP custom_dagSEXP, SEXP theta_optsSEXP, SEXP Sigma_startSEXP, SEXP mvreg_B_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP sample_iwishSEXP, SEXP sample_mvrSEXP, SEXP sample_theta_gibbsSEXP, SEXP upd_theta_optsSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type custom_dag(custom_dagSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_opts(theta_optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma_start(Sigma_startSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mvreg_B_start(mvreg_B_startSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< bool >::type sample_iwish(sample_iwishSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_mvr(sample_mvrSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_theta_gibbs(sample_theta_gibbsSEXP);
    Rcpp::traits::input_parameter< bool >::type upd_theta_opts(upd_theta_optsSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_wishart(Y, X, coords, custom_dag, theta_opts, Sigma_start, mvreg_B_start, mcmc, print_every, sample_iwish, sample_mvr, sample_theta_gibbs, upd_theta_opts, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_latent
Rcpp::List spiox_latent(const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& custom_dag, arma::mat theta_opts, const arma::mat& Sigma_start, const arma::mat& mvreg_B_start, int mcmc, int print_every, bool sample_iwish, bool sample_mvr, bool sample_theta_gibbs, bool upd_theta_opts, int num_threads, int sampling);
RcppExport SEXP _spiox_spiox_latent(SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP custom_dagSEXP, SEXP theta_optsSEXP, SEXP Sigma_startSEXP, SEXP mvreg_B_startSEXP, SEXP mcmcSEXP, SEXP print_everySEXP, SEXP sample_iwishSEXP, SEXP sample_mvrSEXP, SEXP sample_theta_gibbsSEXP, SEXP upd_theta_optsSEXP, SEXP num_threadsSEXP, SEXP samplingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type custom_dag(custom_dagSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_opts(theta_optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma_start(Sigma_startSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mvreg_B_start(mvreg_B_startSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< bool >::type sample_iwish(sample_iwishSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_mvr(sample_mvrSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_theta_gibbs(sample_theta_gibbsSEXP);
    Rcpp::traits::input_parameter< bool >::type upd_theta_opts(upd_theta_optsSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< int >::type sampling(samplingSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_latent(Y, X, coords, custom_dag, theta_opts, Sigma_start, mvreg_B_start, mcmc, print_every, sample_iwish, sample_mvr, sample_theta_gibbs, upd_theta_opts, num_threads, sampling));
    return rcpp_result_gen;
END_RCPP
}
// spiox_predict
Rcpp::List spiox_predict(const arma::mat& X_new, const arma::mat& coords_new, const arma::mat& Y, const arma::mat& X, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::cube& B, const arma::cube& Sigma, const arma::cube& theta, int num_threads);
RcppExport SEXP _spiox_spiox_predict(SEXP X_newSEXP, SEXP coords_newSEXP, SEXP YSEXP, SEXP XSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP BSEXP, SEXP SigmaSEXP, SEXP thetaSEXP, SEXP num_threadsSEXP) {
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
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_predict(X_new, coords_new, Y, X, coords, dag, B, Sigma, theta, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// spiox_latent_predict
Rcpp::List spiox_latent_predict(const arma::mat& X_new, const arma::mat& coords_new, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::cube& W, const arma::cube& B, const arma::cube& Sigma, const arma::mat& Dvec, const arma::cube& theta, int num_threads);
RcppExport SEXP _spiox_spiox_latent_predict(SEXP X_newSEXP, SEXP coords_newSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP WSEXP, SEXP BSEXP, SEXP SigmaSEXP, SEXP DvecSEXP, SEXP thetaSEXP, SEXP num_threadsSEXP) {
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
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(spiox_latent_predict(X_new, coords_new, coords, dag, W, B, Sigma, Dvec, theta, num_threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_spiox_Correlationc", (DL_FUNC) &_spiox_Correlationc, 5},
    {"_spiox_daggp_build", (DL_FUNC) &_spiox_daggp_build, 8},
    {"_spiox_dl_update_variances", (DL_FUNC) &_spiox_dl_update_variances, 3},
    {"_spiox_iox", (DL_FUNC) &_spiox_iox, 9},
    {"_spiox_sfact", (DL_FUNC) &_spiox_sfact, 5},
    {"_spiox_rvec", (DL_FUNC) &_spiox_rvec, 5},
    {"_spiox_make_ix", (DL_FUNC) &_spiox_make_ix, 2},
    {"_spiox_iox_mat", (DL_FUNC) &_spiox_iox_mat, 6},
    {"_spiox_make_candidates", (DL_FUNC) &_spiox_make_candidates, 4},
    {"_spiox_neighbor_search_testset", (DL_FUNC) &_spiox_neighbor_search_testset, 3},
    {"_spiox_S_to_Sigma", (DL_FUNC) &_spiox_S_to_Sigma, 1},
    {"_spiox_S_to_Q", (DL_FUNC) &_spiox_S_to_Q, 1},
    {"_spiox_run_spf_model", (DL_FUNC) &_spiox_run_spf_model, 10},
    {"_spiox_spiox_wishart", (DL_FUNC) &_spiox_spiox_wishart, 14},
    {"_spiox_spiox_latent", (DL_FUNC) &_spiox_spiox_latent, 15},
    {"_spiox_spiox_predict", (DL_FUNC) &_spiox_spiox_predict, 10},
    {"_spiox_spiox_latent_predict", (DL_FUNC) &_spiox_spiox_latent_predict, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_spiox(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
