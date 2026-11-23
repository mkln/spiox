// IW-VI: non-conjugate refinement of the latent mean-field VI fit for (Sigma, Ddiag), Theta fixed.
//
// Mean-field VI (SpIOX::latent_vi) factorises q(W) q(Sigma) q(tau2). Its optimal q(Sigma) is
// IW(Sigma_UQ, n + q): the df is tied to n as if W were observed, so Sigma and Ddiag come out
// over-confident. Here (B, W) are integrated out instead, and the family is
//     q(Sigma, tau2) = IW(Psi, nu) x prod_j IG(alpha_j, beta_j),
// with nu and alpha learned, maximising the ELBO of the marginal posterior p(Sigma, tau2 | Y).
// Each iteration:
//   1. draw (Sigma, tau2) ~ q by reparameterisation (Bartlett for Sigma with the chi-square
//      diagonal by inverse CDF of fixed uniforms, inverse CDF for tau2);
//   2. draw (B, W) ~ p(B, W | Y, Sigma, tau2) exactly with the joint PCG sampler;
//   3. by Fisher's identity, grad log p(Y, W, Sigma, tau2) at that draw is an unbiased estimate of
//      grad log p(Sigma, tau2 | Y); chain rule through the reparameterisation, plus the entropy.
// Adam on the unconstrained parameters; the returned fit averages the iterates over the second half.
// Priors as in update_Sigma_vi / update_Ddiag_vi: Sigma ~ IW(I, q), tau2_j ~ IG(2, 1).

#include "spiox.h"

#include "interrupt.h"
#include <chrono>

namespace iwvi {

const double a0 = 2.0, b0 = 1.0;

// lambda = ( log diag L [q], below-diagonal L by column [q(q-1)/2], log(nu - q - 1) [1],
//            log(alpha - 1) [q], log m [q] ),
// with M = E[Sigma] = L L^T, m_j = E[tau2_j], Psi = (nu - q - 1) M, beta_j = (alpha_j - 1) m_j.
// Parameterising by the means decouples location from df / shape.
struct Par {
  arma::mat L;
  double nu, c;                    // c = nu - q - 1
  arma::vec alpha, m, beta;
};

inline int n_lambda(int q){ return q + q * (q - 1) / 2 + 1 + 2 * q; }
inline int i_nu(int q){ return q + q * (q - 1) / 2; }       // index of log(nu - q - 1)

Par unpack(const arma::vec& lam, int q){
  Par p;
  p.L = arma::zeros(q, q);
  int k = q;
  for(int j = 0; j < q; j++) p.L(j, j) = std::exp(lam(j));
  for(int j = 0; j < q; j++) for(int i = j + 1; i < q; i++) p.L(i, j) = lam(k++);
  p.c = std::exp(lam(k++));
  p.nu = q + 1.0 + p.c;
  p.alpha = 1.0 + arma::exp(lam.subvec(k, k + q - 1)); k += q;
  p.m = arma::exp(lam.subvec(k, k + q - 1));
  p.beta = (p.alpha - 1.0) % p.m;
  return p;
}

arma::vec pack(const arma::mat& M, double nu, const arma::vec& alpha, const arma::vec& beta){
  int q = M.n_rows;
  arma::mat L = arma::chol(arma::symmatu(M), "lower");
  arma::vec lam(n_lambda(q));
  int k = q;
  for(int j = 0; j < q; j++) lam(j) = std::log(L(j, j));
  for(int j = 0; j < q; j++) for(int i = j + 1; i < q; i++) lam(k++) = L(i, j);
  lam(k++) = std::log(nu - q - 1.0);
  lam.subvec(k, k + q - 1) = arma::log(alpha - 1.0); k += q;
  lam.subvec(k, k + q - 1) = arma::log(beta / (alpha - 1.0));
  return lam;
}

// Bartlett factor: lower-triangular T with T T^T ~ Wishart(I, nu). Diagonal T_ii^2 ~ chi^2_{nu - i}
// (i = 0..q-1) by inverse CDF of fixed uniforms u, so T is a smooth function of nu; below-diagonal
// entries are the fixed standard normals in Z.
arma::mat bartlett_T(double nu, const arma::vec& u, const arma::mat& Z){
  int q = u.n_elem;
  arma::mat T = arma::zeros(q, q);
  for(int i = 0; i < q; i++){
    T(i, i) = std::sqrt(R::qgamma(u(i), 0.5 * (nu - i), 2.0, 1, 0));
    for(int j = 0; j < i; j++) T(i, j) = Z(i, j);
  }
  return T;
}

// Sigma = c L (T T^T)^{-1} L^T, i.e. Sigma^{-1} = L^{-T} T T^T L^{-1} / c ~ Wishart(Psi^{-1}, nu).
// Returns Sigma; Binv_out = (T T^T)^{-1}.
arma::mat sigma_from(const Par& p, const arma::mat& T, arma::mat& Binv_out){
  arma::mat Tinv = arma::inv(arma::trimatl(T));
  Binv_out = Tinv.t() * Tinv;
  arma::mat S = p.c * p.L * Binv_out * p.L.t();
  return 0.5 * (S + S.t());
}

double lmvgamma(double a, int q){
  double s = 0.25 * q * (q - 1) * std::log(M_PI);
  for(int i = 0; i < q; i++) s += R::lgammafn(a - 0.5 * i);
  return s;
}

// entropy of IW(Psi, nu), as a function of nu and log|Psi|
double h_iw(double nu, double ldP, int q){
  double sdig = 0.0;
  for(int i = 0; i < q; i++) sdig += R::digamma(0.5 * (nu - i));
  double Elogdet = ldP - q * std::log(2.0) - sdig;          // E[log|Sigma|]
  return -0.5 * nu * ldP + 0.5 * nu * q * std::log(2.0) + lmvgamma(0.5 * nu, q) +
    0.5 * (nu + q + 1.0) * Elogdet + 0.5 * nu * q;
}

// entropy of IG(alpha, beta)
double h_ig(double alpha, double beta){
  return alpha + std::log(beta) + R::lgammafn(alpha) - (1.0 + alpha) * R::digamma(alpha);
}

inline double logdetP(const Par& p, int q){
  return q * std::log(p.c) + 2.0 * arma::accu(arma::log(p.L.diag()));
}

inline double clamp01(double x){ return std::min(std::max(x, 1e-10), 1.0 - 1e-10); }

// One reparameterised draw (Sigma, tau2) ~ q given the noise (Z, u, ut).
struct Draw { arma::mat Sigma, Q, Binv; arma::vec tau2; };

Draw draw_q(const Par& p, const arma::mat& Z, const arma::vec& u, const arma::vec& ut){
  int q = u.n_elem;
  Draw d;
  arma::mat T = bartlett_T(p.nu, u, Z);
  d.Sigma = sigma_from(p, T, d.Binv);
  arma::mat A = T.t() * arma::inv(arma::trimatl(p.L));
  d.Q = arma::symmatu(A.t() * A / p.c);                        // Sigma^{-1} without inverting Sigma
  d.tau2 = arma::vec(q);
  for(int j = 0; j < q; j++) d.tau2(j) = p.beta(j) / R::qgamma(ut(j), p.alpha(j), 1.0, 1, 0);
  return d;
}

// ELBO gradient in lambda for one draw, given G = d/dSigma and gt = d/dtau2 of log p(Y, W, Sigma, tau2).
arma::vec grad_lambda(const arma::vec& lam, const Par& p, const Draw& d, const arma::mat& Z,
                      const arma::vec& u, const arma::vec& ut, const arma::mat& G, const arma::vec& gt,
                      double h = 1e-5){
  int q = u.n_elem, kn = i_nu(q);
  arma::vec gr = arma::zeros(n_lambda(q));
  // L: dSigma = c (dL Binv L^T + L Binv dL^T)  =>  d<G, Sigma>/dL = 2 c G L Binv (G symmetric).
  // The entropy depends on L only through log|Psi| = q log c + 2 sum log L_jj, with coefficient (q + 1)/2.
  arma::mat gL = 2.0 * p.c * G * p.L * d.Binv;
  for(int j = 0; j < q; j++) gr(j) = gL(j, j) * p.L(j, j) + (q + 1.0);
  int k = q;
  for(int j = 0; j < q; j++) for(int i = j + 1; i < q; i++) gr(k++) = gL(i, j);
  // nu: central finite difference in log(nu - q - 1) with the same noise (Z, u)
  {
    arma::vec lp = lam, lm = lam; lp(kn) += h; lm(kn) -= h;
    Par pp = unpack(lp, q), pm = unpack(lm, q);
    arma::mat Bp, Bm;
    arma::mat Sp = sigma_from(pp, bartlett_T(pp.nu, u, Z), Bp);
    arma::mat Sm = sigma_from(pm, bartlett_T(pm.nu, u, Z), Bm);
    gr(kn) = arma::accu(G % (Sp - Sm)) / (2.0 * h) +
      (h_iw(pp.nu, logdetP(pp, q), q) - h_iw(pm.nu, logdetP(pm, q), q)) / (2.0 * h);
  }
  for(int j = 0; j < q; j++){
    // alpha_j: central finite difference in log(alpha_j - 1), m_j fixed, same uniform ut_j
    double ap = 1.0 + std::exp(lam(kn + 1 + j) + h), amn = 1.0 + std::exp(lam(kn + 1 + j) - h);
    double bp = (ap - 1.0) * p.m(j), bm = (amn - 1.0) * p.m(j);
    double tp = bp / R::qgamma(ut(j), ap, 1.0, 1, 0), tm = bm / R::qgamma(ut(j), amn, 1.0, 1, 0);
    gr(kn + 1 + j) = gt(j) * (tp - tm) / (2.0 * h) + (h_ig(ap, bp) - h_ig(amn, bm)) / (2.0 * h);
    // log m_j: tau2_j is proportional to m_j; the entropy has + log beta_j
    gr(kn + 1 + q + j) = gt(j) * d.tau2(j) + 1.0;
  }
  return gr;
}

} // namespace iwvi


Rcpp::List latent_iwvi(SpIOX& m, int iters, double lr, double lr_df, int print_every, int pred_smp){
  using namespace iwvi;
  const int q = m.q;
  const double n = m.n;
  const int D = n_lambda(q), kn = i_nu(q);
  
  arma::vec nj(q);
  for(int j = 0; j < q; j++) nj(j) = m.avail_by_outcome(j).n_elem;
  
  // start at the mean-field solution: IW(Sigma_UQ, n + q) x prod_j IG(nj/2 + a0, b0 + Ddiag_UQ/2)
  arma::vec al0 = nj / 2.0 + a0;
  arma::vec be0 = b0 + 0.5 * m.Ddiag_UQ;
  arma::vec lam = pack(m.Sigma_UQ / (n - 1.0), n + q, al0, be0);
  
  // Adam state; per-coordinate base step sizes (lr_df on log(nu - q - 1) and log(alpha - 1))
  arma::vec lrv(D); lrv.fill(lr);
  lrv(kn) = lr_df;
  lrv.subvec(kn + 1, kn + q).fill(lr_df);
  arma::vec am = arma::zeros(D), av = arma::zeros(D), lam_bar = arma::zeros(D);
  const double b1 = 0.9, b2 = 0.999;
  int n_avg = 0;
  
  // same PC choice as latent_vi()
  const SpIOX::PrecondChoice use_pc = (m.precond_choice == SpIOX::PRECOND_AUTO) ? 
              SpIOX::PRECOND_VADU : m.precond_choice;
  
  arma::vec df_ratio_trace(iters);
  arma::uvec cg_trace(iters);
  auto t0 = std::chrono::steady_clock::now();
  
  if(print_every > 0) Rcpp::Rcout << "Starting IW-VI (" << iters << " iterations)." << std::endl;
  
  for(int t = 1; t <= iters; t++){
    Par p = unpack(lam, q);
    
    // 1. reparameterised draw (Sigma, tau2) ~ q
    arma::mat Z = arma::randn(q, q);
    arma::vec u(q), ut(q);
    for(int i = 0; i < q; i++){ u(i) = clamp01(R::runif(0.0, 1.0)); ut(i) = clamp01(R::runif(0.0, 1.0)); }
    Draw d = draw_q(p, Z, u, ut);
    
    // 2. exact conditional draw (B, W) | Y, Sigma, tau2
    m.Sigma = d.Sigma;
    m.Q = d.Q;
    m.Si = arma::chol(d.Q, "lower");
    m.S = arma::inv(arma::trimatl(m.Si));
    m.Ddiag = d.tau2;
    int cg_iter = 0;
    m.gibbs_BW_block(cg_iter, use_pc, true);
    m.compute_V();                                               // V = [H_j W_j]
    arma::mat VTV = m.V.t() * m.V;
    arma::mat E = m.Y - m.X * m.B - m.W;
    arma::vec SS(q);
    for(int j = 0; j < q; j++){
      arma::vec ej = E.col(j);
      SS(j) = arma::accu(arma::square(ej(m.avail_by_outcome(j))));
    }
    
    // 3. gradients of log p(Y, W, Sigma, tau2) in Sigma (symmetric, unconstrained) and tau2,
    //    then chain rule to lambda
    arma::mat G = -0.5 * (n + 2.0 * q + 1.0) * d.Q + 0.5 * d.Q * (VTV + arma::eye(q, q)) * d.Q;
    arma::vec gt = -(nj / 2.0 + a0 + 1.0) / d.tau2 + (b0 + 0.5 * SS) / arma::square(d.tau2);
    arma::vec gr = grad_lambda(lam, p, d, Z, u, ut, G, gt);
    
    // Adam ascent step with decaying step size
    am = b1 * am + (1.0 - b1) * gr;
    av = b2 * av + (1.0 - b2) * arma::square(gr);
    arma::vec step = (lrv / std::sqrt(1.0 + t / 200.0)) %
      (am / (1.0 - std::pow(b1, t))) / (arma::sqrt(av / (1.0 - std::pow(b2, t))) + 1e-8);
    lam += step;
    
    if(t > iters / 2){ lam_bar += lam; n_avg++; }
    df_ratio_trace(t - 1) = std::exp(lam(kn)) / (n - 1.0);
    cg_trace(t - 1) = cg_iter;
    
    if(print_every > 0 && t % print_every == 0){
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Rcpp::Rcout << "IW-VI iteration: " << t << "  (CG: " << cg_iter << " iters)  df ratio "
                  << df_ratio_trace(t - 1) << "  " << el << "s" << std::endl;
    }
    if(checkInterrupt()) Rcpp::stop("Interrupted by the user.");
  }
  
  Par f = unpack(lam_bar / (double) std::max(n_avg, 1), q);
  arma::mat M = f.L * f.L.t();
  
  
  // predictive samples at the missing entries: (Sigma, tau2) ~ q, then (B, W) | Y, Sigma, tau2 exactly
  const arma::uvec& na = m.Y_na_indices;
  arma::mat Y_missing_samples(na.n_elem, pred_smp);
  for(int s = 0; s < pred_smp; s++){
    arma::mat Z = arma::randn(q, q); arma::vec u(q), ut(q);
    for(int i = 0; i < q; i++){ u(i) = clamp01(R::runif(0.0, 1.0)); ut(i) = clamp01(R::runif(0.0, 1.0)); }
    Draw d = draw_q(f, Z, u, ut);
    m.Sigma = d.Sigma; m.Q = d.Q; m.Si = arma::chol(d.Q, "lower"); m.S = arma::inv(arma::trimatl(m.Si)); m.Ddiag = d.tau2;
    int cg_iter = 0;
    m.gibbs_BW_block(cg_iter, use_pc, true);
    arma::mat noise = arma::randn(m.n, q); noise.each_row() %= arma::sqrt(d.tau2).t();
    arma::mat Yhat = m.X * m.B + m.W + noise;
    Y_missing_samples.col(s) = Yhat(na);
    if(checkInterrupt()) Rcpp::stop("Interrupted by the user.");
  }
  
  return Rcpp::List::create(
    Rcpp::Named("Psi") = f.c * M,
    Rcpp::Named("nu") = f.nu,
    Rcpp::Named("alpha") = f.alpha,
    Rcpp::Named("beta") = f.beta,
    Rcpp::Named("Sigma_mean") = M,
    Rcpp::Named("Ddiag_mean") = f.m,
    Rcpp::Named("df_ratio") = f.c / (n - 1.0),
    Rcpp::Named("df_ratio_trace") = df_ratio_trace,
    Rcpp::Named("cg_iter") = cg_trace,
    Rcpp::Named("iters") = iters,
    Rcpp::Named("Y_missing_samples") = Y_missing_samples,
    Rcpp::Named("time") = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
  );
}