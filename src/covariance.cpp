#include "covariance.h"

using namespace std;


// helper: squared distance between two rows i and j of coords
static inline double sqdist_rows(const arma::mat& coords, arma::uword i, arma::uword j) {
  const arma::uword n = coords.n_rows;
  const arma::uword D = coords.n_cols;
  
  const double* base = coords.memptr(); // column-major
  double ss = 0.0;
  for (arma::uword d = 0; d < D; ++d) {
    const double xi = base[i + d * n];
    const double xj = base[j + d * n];
    const double diff = xi - xj;
    ss += diff * diff;
  }
  return ss;
}

// matern
void matern_inplace(arma::mat& res,
                    const arma::mat& coords,
                    const arma::uvec& ix, const arma::uvec& iy,
                    const double& phi, const double& nu,
                    const double& sigmasq,
                    double* bessel_ws,
                    const double& alpha = 0, bool same = false) {
  
  // half-integer checks
  const bool nu05 = std::abs(nu - 0.5) < 1e-12;
  const bool nu15 = std::abs(nu - 1.5) < 1e-12;
  const bool nu25 = std::abs(nu - 2.5) < 1e-12;
  
  // spatial variance portion (off-diagonal)
  const double s = (1.0 - alpha) * sigmasq;
  
  // --- fast path for half-integers ---
  if (nu05 || nu15 || nu25) {
    
    auto matern_halfint_from_r = [&](double r) -> double {
      // Keep your diagonal behavior exactly:
      // when r==0, return sigmasq (not s, not sigmasq*(1-alpha), etc.)
      if (r <= 0.0) return sigmasq;
      
      const double e = std::exp(-r);
      
      if (nu05) {
        // exp(-r)
        return s * e;
      } else if (nu15) {
        // (1 + r) exp(-r)
        return s * (1.0 + r) * e;
      } else {
        // (1 + r + r^2/3) exp(-r)
        const double r2 = r * r;
        return s * (1.0 + r + (r2 / 3.0)) * e;
      }
    };
    
    if (same) {
      for (arma::uword a = 0; a < ix.n_rows; ++a) {
        const arma::uword ia = ix(a);
        for (arma::uword b = a; b < iy.n_rows; ++b) {
          const arma::uword ib = iy(b);
          
          // squared distance -> r = phi * sqrt(ss)
          const double ss = sqdist_rows(coords, ia, ib);
          const double r  = (ss > 0.0) ? (phi * std::sqrt(ss)) : 0.0;
          
          res(a, b) = matern_halfint_from_r(r);
        }
      }
      res = arma::symmatu(res);
    } else {
      for (arma::uword a = 0; a < ix.n_rows; ++a) {
        const arma::uword ia = ix(a);
        for (arma::uword b = 0; b < iy.n_rows; ++b) {
          const arma::uword ib = iy(b);
          
          const double ss = sqdist_rows(coords, ia, ib);
          const double r  = (ss > 0.0) ? (phi * std::sqrt(ss)) : 0.0;
          
          res(a, b) = matern_halfint_from_r(r);
        }
      }
    }
    
    return;
  }
  
  // --- fallback: original general-nu Matérn (your code, but with the same sqdist optimization) ---
  int threadid = 0;
#ifdef _OPENMP
  threadid = omp_get_thread_num();
#endif
  
  const double pref = (1.0 - alpha) * sigmasq * std::pow(2.0, 1.0 - nu) / R::gammafn(nu);
  
  if (same) {
    for (arma::uword a = 0; a < ix.n_rows; ++a) {
      const arma::uword ia = ix(a);
      for (arma::uword b = a; b < iy.n_rows; ++b) {
        const arma::uword ib = iy(b);
        
        const double ss = sqdist_rows(coords, ia, ib);
        const double r  = (ss > 0.0) ? (phi * std::sqrt(ss)) : 0.0;
        
        if (r > 0.0) {
          res(a, b) = std::pow(r, nu) * pref *
            R::bessel_k_ex(r, nu, 1.0, &bessel_ws[threadid * MAT_NU_MAX]);
        } else {
          res(a, b) = sigmasq;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for (arma::uword a = 0; a < ix.n_rows; ++a) {
      const arma::uword ia = ix(a);
      for (arma::uword b = 0; b < iy.n_rows; ++b) {
        const arma::uword ib = iy(b);
        
        const double ss = sqdist_rows(coords, ia, ib);
        const double r  = (ss > 0.0) ? (phi * std::sqrt(ss)) : 0.0;
        
        if (r > 0.0) {
          res(a, b) = std::pow(r, nu) * pref *
            R::bessel_k_ex(r, nu, 1.0, &bessel_ws[threadid * MAT_NU_MAX]);
        } else {
          res(a, b) = sigmasq;
        }
      }
    }
  }
}

static void matern_grad_halfint_inplace(std::vector<arma::mat>& dres,
                                        const arma::mat& coords,
                                        const arma::uvec& ix, const arma::uvec& iy,
                                        const double phi,
                                        const double nu,          // must be 0.5 or 1.5
                                        const double sigmasq,
                                        const double alpha,
                                        bool same) {
  // dres.size() must be 4, each sized (ix.n_rows x iy.n_rows)
  for (auto& M : dres) M.zeros(ix.n_rows, iy.n_rows);
  
  const double s = (1.0 - alpha) * sigmasq;
  
  const bool nu05 = std::abs(nu - 0.5) < 1e-12;
  const bool nu15 = std::abs(nu - 1.5) < 1e-12;
  if (!(nu05 || nu15)) {
    Rcpp::stop("matern_grad_halfint_inplace: nu must be 0.5 or 1.5");
  }
  
  auto fill_pair = [&](arma::uword a, arma::uword b, arma::uword ia, arma::uword ib) {
    const double ss = sqdist_rows(coords, ia, ib);
    
    if (ss <= 0.0) {
      // diagonal / identical locations: covariance = sigmasq (per your matern_inplace)
      dres[0](a,b) = 0.0; // d/dphi
      dres[1](a,b) = 1.0; // d/dsigmasq
      dres[2](a,b) = 0.0; // d/dnu (unused here)
      dres[3](a,b) = 0.0; // d/dalpha
      return;
    }
    
    const double d = std::sqrt(ss);
    const double r = phi * d;
    const double e = std::exp(-r);
    
    if (nu05) {
      // C = s * e
      const double C = s * e;
      
      // dC/dphi = s * (-d) * e = -d * C
      dres[0](a,b) = -d * C;
      
      // dC/dsigmasq = (1-alpha) * e
      dres[1](a,b) = (1.0 - alpha) * e;
      
      // dC/dnu = 0 in this specialized routine
      dres[2](a,b) = 0.0;
      
      // dC/dalpha = -sigmasq * e
      dres[3](a,b) = -sigmasq * e;
      
    } else {
      // nu = 1.5
      // C = s * (1+r) * e
      const double f = (1.0 + r) * e;
      const double C = s * f;
      
      // df/dr = -r * e  (since d/dr[(1+r)e^{-r}] = -r e^{-r})
      // dC/dphi = s * df/dr * dr/dphi = s * (-r e) * d = -s * d * r * e
      dres[0](a,b) = -s * d * r * e;
      
      // dC/dsigmasq = (1-alpha) * f
      dres[1](a,b) = (1.0 - alpha) * f;
      
      // dC/dnu = 0 here
      dres[2](a,b) = 0.0;
      
      // dC/dalpha = -sigmasq * f
      dres[3](a,b) = -sigmasq * f;
    }
  };
  
  if (same) {
    for (arma::uword a = 0; a < ix.n_rows; ++a) {
      const arma::uword ia = ix(a);
      for (arma::uword b = a; b < iy.n_rows; ++b) {
        const arma::uword ib = iy(b);
        fill_pair(a, b, ia, ib);
      }
    }
    for (auto& M : dres) M = arma::symmatu(M);
  } else {
    for (arma::uword a = 0; a < ix.n_rows; ++a) {
      const arma::uword ia = ix(a);
      for (arma::uword b = 0; b < iy.n_rows; ++b) {
        const arma::uword ib = iy(b);
        fill_pair(a, b, ia, ib);
      }
    }
  }
}

// Wrapper returning vector<mat> like your earlier Correlationf_grad
std::vector<arma::mat>
  Correlationf_grad_matern_halfint(const arma::mat& coords,
                                   const arma::uvec& ix, const arma::uvec& iy,
                                   const arma::vec& theta,
                                   bool same) {
    if (theta.n_elem < 4) Rcpp::stop("Expected theta = (phi,sigmasq,nu,alpha).");
    
    const double phi     = theta(0);
    const double sigmasq = theta(1);
    const double nu      = theta(2);
    const double alpha   = theta(3);
    
    std::vector<arma::mat> dres(4);
    for (auto& M : dres) M.set_size(ix.n_rows, iy.n_rows);
    
    matern_grad_halfint_inplace(dres, coords, ix, iy, phi, nu, sigmasq, alpha, same);
    return dres;
  }

// powered exponential nu<2
void powerexp_inplace(arma::mat& res, 
                      const arma::mat& coords,
                      const arma::uvec& ix, const arma::uvec& iy, 
                      const double& phi, const double& nu, 
                      const double& sigmasq, const double& alpha,
                      bool same){
  
  // 1-alpha is the proportion of variance explained by spatial component
  // alpha is is the proportion explained by nugget effect
  
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hh = arma::norm(delta);
        if(hh==0){
          res(i, j) = sigmasq; // so the nugget is alpha*sigmasq
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq * (1-alpha);
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=0; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hh = arma::norm(delta);
        if(hh==0){
          res(i, j) = sigmasq;
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq * (1-alpha);
        }
      }
    }
  }
}


inline double matern_corr_halfint(double rphi, double nu) {
  // rphi = ||s-s'|| * phi
  // returns Matérn correlation (variance 1, no nugget)
  if (rphi <= 0.0) return 1.0;
  
  if (std::abs(nu - 0.5) < 1e-12) {
    // exp(-r)
    return std::exp(-rphi);
  } else if (std::abs(nu - 1.5) < 1e-12) {
    // (1 + r) exp(-r)
    return (1.0 + rphi) * std::exp(-rphi);
  } else if (std::abs(nu - 2.5) < 1e-12) {
    // (1 + r + r^2/3) exp(-r)
    const double r2 = rphi * rphi;
    return (1.0 + rphi + r2 / 3.0) * std::exp(-rphi);
  } else {
    Rcpp::stop("matern_corr_halfint: nu must be 0.5, 1.5, or 2.5.");
  }
  return 0.0; // never reached
}

void sqexpcovariates_inplace(arma::mat& res,
                             const arma::mat& cx,                 // first 2 cols spatial, rest features
                             const arma::uvec& ix, const arma::uvec& iy,
                             const double phi,                    // ONLY free spatial parameter
                             const double nu_fixed,               // must be 0.5,1.5,2.5
                             const arma::mat& Theta_x,            // D_x x k projection
                             const double alpha,                  // off-diagonal shrink
                             bool same){
  
  const arma::uword D = cx.n_cols;
  if (D < 2) Rcpp::stop("sqexpcovariates_inplace: cx must have at least 2 spatial columns.");
  const arma::uword D_x = D - 2;
  
  if (phi < 0.0) Rcpp::stop("sqexpcovariates_inplace: phi must be >= 0.");
  const bool ok_nu =
    (std::abs(nu_fixed - 0.5) < 1e-12) ||
    (std::abs(nu_fixed - 1.5) < 1e-12) ||
    (std::abs(nu_fixed - 2.5) < 1e-12);
  if (!ok_nu) Rcpp::stop("sqexpcovariates_inplace: nu_fixed must be in {0.5, 1.5, 2.5}.");
  
  if (!(alpha >= 0.0 && alpha <= 1.0)) Rcpp::stop("sqexpcovariates_inplace: alpha must be in [0,1].");
  
  if (D_x == 0) {
    // no features: Theta_x can be 0xk; still allow
    if (Theta_x.n_rows != 0) Rcpp::stop("sqexpcovariates_inplace: Theta_x.n_rows must be 0 when D_x=0.");
  } else {
    if (Theta_x.n_rows != D_x) Rcpp::stop("sqexpcovariates_inplace: Theta_x.n_rows must equal cx.n_cols-2.");
    if (Theta_x.n_cols < 1)    Rcpp::stop("sqexpcovariates_inplace: Theta_x must have at least 1 col (k>=1).");
  }
  
  const double off_scale = 1.0 - alpha;
  
  auto feature_factor = [&](const arma::rowvec& Xi, const arma::rowvec& Xj) -> double {
    if (D_x == 0) return 1.0;
    const arma::rowvec dx = Xi.cols(2, D-1) - Xj.cols(2, D-1); // 1 x D_x
    const arma::rowvec z  = dx * Theta_x;                      // 1 x k
    const double qx = arma::dot(z, z);                          // ||z||^2
    return std::exp(-qx);
  };
  
  auto spatial_factor = [&](const arma::rowvec& Xi, const arma::rowvec& Xj) -> double {
    const double ds0 = Xi(0) - Xj(0);
    const double ds1 = Xi(1) - Xj(1);
    const double r   = std::sqrt(ds0*ds0 + ds1*ds1);
    return matern_corr_halfint(r * phi, nu_fixed);
  };
  
  if (same) {
    for (arma::uword i = 0; i < ix.n_rows; ++i) {
      const arma::rowvec Xi = cx.row(ix(i));
      for (arma::uword j = i; j < iy.n_rows; ++j) {
        if (ix(i) == iy(j)) {
          res(i,j) = 1.0; // exact diagonal
        } else {
          const arma::rowvec Xj = cx.row(iy(j));
          res(i,j) = off_scale * spatial_factor(Xi, Xj) * feature_factor(Xi, Xj);
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for (arma::uword i = 0; i < ix.n_rows; ++i) {
      const arma::rowvec Xi = cx.row(ix(i));
      for (arma::uword j = 0; j < iy.n_rows; ++j) {
        if (ix(i) == iy(j)) {
          res(i,j) = 1.0;
        } else {
          const arma::rowvec Xj = cx.row(iy(j));
          res(i,j) = off_scale * spatial_factor(Xi, Xj) * feature_factor(Xi, Xj);
        }
      }
    }
  }
}

arma::mat Correlationf(
    const arma::mat& coords,
    const arma::uvec& ix, const arma::uvec& iy,
    const arma::vec& theta,
    double * bessel_ws,
    int matern,
    bool same){
  
  arma::mat res = arma::zeros(ix.n_rows, iy.n_rows);
  
  if (matern == 3) {
    const arma::uword D = coords.n_cols;
    if (D < 2) Rcpp::stop("Correlationf (matern=3): coords must have at least 2 spatial columns.");
    const arma::uword D_x = D - 2;
    
    // theta layout for matern==3:
    //   theta[0]           = phi (>=0)
    //   theta[1:(end-1)]   = vec(Theta_x), Theta_x is D_x x k (col-major)
    //   theta[end]         = alpha in [0,1]
    if (theta.n_elem < 2) {
      Rcpp::stop("Correlationf (matern=3): theta must have at least 2 elements: phi and alpha.");
    }
    
    const double phi   = theta(0);
    const double alpha = theta(theta.n_elem - 1);
    
    if (phi < 0.0) Rcpp::stop("Correlationf (matern=3): phi must be >= 0.");
    if (!(alpha >= 0.0 && alpha <= 1.0)) Rcpp::stop("Correlationf (matern=3): alpha must be in [0,1].");
    
    const double nu_fixed = 1.5; // choose between 0.5, 1.5, 2.5
    const bool ok_nu =
      (std::abs(nu_fixed - 0.5) < 1e-12) ||
      (std::abs(nu_fixed - 1.5) < 1e-12) ||
      (std::abs(nu_fixed - 2.5) < 1e-12);
    if (!ok_nu) Rcpp::stop("Correlationf (matern=3): NU_FIXED_M3 must be 0.5, 1.5, or 2.5.");
    
    const arma::uword n_x = theta.n_elem - 2; // everything except phi and alpha
    if (D_x == 0) {
      // no features: allow n_x==0; Theta_x is empty
      if (n_x != 0) {
        Rcpp::stop("Correlationf (matern=3): D_x=0 but theta contains feature parameters.");
      }
      arma::mat Theta_x(0, 1, arma::fill::zeros);
      sqexpcovariates_inplace(res, coords, ix, iy, phi, nu_fixed, Theta_x, alpha, same);
      return res;
    }
    
    if (n_x == 0) {
      Rcpp::stop("Correlationf (matern=3): no entries provided for Theta_x.");
    }
    if (n_x % D_x != 0) {
      Rcpp::stop("Correlationf (matern=3): length(theta[1:(end-1)]) must be divisible by D_x = coords.n_cols-2.");
    }
    const arma::uword k = n_x / D_x;
    if (k < 1) Rcpp::stop("Correlationf (matern=3): inferred k must be >= 1.");
    
    // Wrap Theta_x without copying (column-major)
    arma::mat Theta_x(const_cast<double*>(theta.memptr() + 1), D_x, k,
                      /*copy_aux_mem=*/false, /*strict=*/true);
    
    sqexpcovariates_inplace(res, coords, ix, iy, phi, nu_fixed, Theta_x, alpha, same);
    return res;
    
  } else {
    // ---- original behavior for matern / powerexp / wave ----
    double phi = theta(0);
    double sigmasq = theta(1);
    double nu = theta(2);
    double alpha = theta(3);
    
    if (matern == 1) {
      matern_inplace(res, coords, ix, iy, phi, nu, sigmasq, bessel_ws, alpha, same);
    } else if (matern == 0) {
      powerexp_inplace(res, coords, ix, iy, phi, nu, sigmasq, alpha, same);
    } 
    return res;
  }
}
std::vector<arma::mat>
  Correlationf_grad(const arma::mat& coords,
                    const arma::uvec& ix, const arma::uvec& iy,
                    const arma::vec& theta,
                    double * bessel_ws,
                    const int matern,
                    const bool same)
{
  // Returns vector of matrices dK/dtheta_t with same dims as K(ix,iy).
  // For matern==3: theta = [phi, vec(Theta_x), alpha], nu is FIXED 
  // For matern!=3 (original): theta = [phi, sigmasq, nu, alpha] and currently only supports matern==1 with half-int n.
  
  if (matern == 3) {
    
    const arma::uword D = coords.n_cols;
    if (D < 2) Rcpp::stop("Correlationf_grad (matern=3): coords must have at least 2 spatial columns.");
    const arma::uword D_x = D - 2;
    
    if (theta.n_elem < 2) {
      Rcpp::stop("Correlationf_grad (matern=3): theta must have at least 2 elements: phi and alpha.");
    }
    
    const double phi   = theta(0);
    const double alpha = theta(theta.n_elem - 1);
    
    if (phi < 0.0) Rcpp::stop("Correlationf_grad (matern=3): phi must be >= 0.");
    if (!(alpha >= 0.0 && alpha <= 1.0)) Rcpp::stop("Correlationf_grad (matern=3): alpha must be in [0,1].");
    
    const double nu_fixed = 1.5;
    const bool ok_nu =
      (std::abs(nu_fixed - 0.5) < 1e-12) ||
      (std::abs(nu_fixed - 1.5) < 1e-12) ||
      (std::abs(nu_fixed - 2.5) < 1e-12);
    if (!ok_nu) Rcpp::stop("Correlationf_grad (matern=3): NU_FIXED_M3 must be 0.5, 1.5, or 2.5.");
    
    // layout: [phi, vec(Theta_x), alpha]
    const arma::uword n_x = theta.n_elem - 2;     // number of Theta_x parameters
    arma::uword k = 0;
    
    if (D_x == 0) {
      if (n_x != 0) Rcpp::stop("Correlationf_grad (matern=3): D_x=0 but theta contains feature parameters.");
      k = 1; // dummy
    } else {
      if (n_x == 0) Rcpp::stop("Correlationf_grad (matern=3): no entries provided for Theta_x.");
      if (n_x % D_x != 0) {
        Rcpp::stop("Correlationf_grad (matern=3): length(vec(Theta_x)) must be divisible by D_x=coords.n_cols-2.");
      }
      k = n_x / D_x;
      if (k < 1) Rcpp::stop("Correlationf_grad (matern=3): inferred k must be >= 1.");
    }
    
    // Wrap Theta_x (no copy) if present
    arma::mat Theta_x;
    if (D_x > 0) {
      Theta_x = arma::mat(const_cast<double*>(theta.memptr() + 1), D_x, k,
                          /*copy_aux_mem=*/false, /*strict=*/true);
    } else {
      Theta_x.set_size(0, k);
      Theta_x.zeros();
    }
    
    // total parameters in theta
    const arma::uword ptheta = theta.n_elem;
    
    std::vector<arma::mat> dres(ptheta);
    for (auto &M : dres) M.zeros(ix.n_rows, iy.n_rows);
    
    // helper: Matérn correlation + derivative wrt phi for half-integer nu
    auto matern_corr_and_dphi = [&](double r, double &R, double &dR_dphi) {
      // r = ||s-s'|| (Euclidean); phi >= 0
      if (r <= 0.0) { R = 1.0; dR_dphi = 0.0; return; }
      const double t = r * phi; // rphi
      const double et = std::exp(-t);
      
      if (std::abs(nu_fixed - 0.5) < 1e-12) {
        // R = exp(-t); dR/dphi = -r exp(-t)
        R = et;
        dR_dphi = -r * et;
      } else if (std::abs(nu_fixed - 1.5) < 1e-12) {
        // R = (1+t) e^{-t}; dR/dt = -t e^{-t}; dR/dphi = r * dR/dt = -r t e^{-t}
        R = (1.0 + t) * et;
        dR_dphi = -r * t * et;
      } else {
        // nu=2.5: R = (1+t+t^2/3)e^{-t}; dR/dt = -(t + t^2)/3 e^{-t}
        const double t2 = t * t;
        R = (1.0 + t + t2 / 3.0) * et;
        dR_dphi = -r * (t + t2) / 3.0 * et;
      }
    };
    
    const double off_scale = 1.0 - alpha;
    
    if (same) {
      for (arma::uword a = 0; a < ix.n_rows; ++a) {
        const arma::rowvec Xi = coords.row(ix(a));
        for (arma::uword b = a; b < iy.n_rows; ++b) {
          
          if (ix(a) == iy(b)) {
            // K(i,i)=1 constant -> all derivatives 0
            // already zero
            continue;
          }
          
          const arma::rowvec Xj = coords.row(iy(b));
          
          // ---- spatial part ----
          const double ds0 = Xi(0) - Xj(0);
          const double ds1 = Xi(1) - Xj(1);
          const double r = std::sqrt(ds0*ds0 + ds1*ds1);
          
          double Rs, dRs_dphi;
          matern_corr_and_dphi(r, Rs, dRs_dphi);
          
          // ---- feature part ----
          double Rx = 1.0;
          arma::rowvec dx;
          arma::rowvec z;
          if (D_x > 0) {
            dx = Xi.cols(2, D-1) - Xj.cols(2, D-1); // 1 x D_x
            z  = dx * Theta_x;                      // 1 x k
            const double qx = arma::dot(z, z);
            Rx = std::exp(-qx);
          }
          
          const double base = Rs * Rx;   // without (1-alpha)
          
          // d/d alpha: K = (1-alpha)*base => dK/dalpha = -base
          dres[ptheta - 1](a, b) = -base;
          
          // d/d phi: (1-alpha) * (dRs/dphi) * Rx
          dres[0](a, b) = off_scale * dRs_dphi * Rx;
          
          // d/d Theta_x entries: (1-alpha) * Rs * dRx/dTheta
          // Rx = exp(-||dx*Theta_x||^2) = exp(-sum_l z_l^2), z = dx*Theta_x
          // dRx/dTheta_{p,l} = Rx * (-2 z_l) * dx_p
          if (D_x > 0) {
            for (arma::uword l = 0; l < k; ++l) {
              const double zl = z(l);
              const double coeff = off_scale * Rs * Rx * (-2.0 * zl);
              // parameters are packed column-major: Theta_x has D_x rows, k cols,
              // and start at theta index 1.
              const arma::uword col_offset = 1 + l * D_x;
              for (arma::uword p = 0; p < D_x; ++p) {
                dres[col_offset + p](a, b) = coeff * dx(p);
              }
            }
          }
        }
      }
      // symmetrize all gradient matrices
      for (auto &G : dres) G = arma::symmatu(G);
      
    } else {
      for (arma::uword a = 0; a < ix.n_rows; ++a) {
        const arma::rowvec Xi = coords.row(ix(a));
        for (arma::uword b = 0; b < iy.n_rows; ++b) {
          
          if (ix(a) == iy(b)) {
            continue;
          }
          
          const arma::rowvec Xj = coords.row(iy(b));
          
          const double ds0 = Xi(0) - Xj(0);
          const double ds1 = Xi(1) - Xj(1);
          const double r = std::sqrt(ds0*ds0 + ds1*ds1);
          
          double Rs, dRs_dphi;
          matern_corr_and_dphi(r, Rs, dRs_dphi);
          
          double Rx = 1.0;
          arma::rowvec dx;
          arma::rowvec z;
          if (D_x > 0) {
            dx = Xi.cols(2, D-1) - Xj.cols(2, D-1);
            z  = dx * Theta_x;
            const double qx = arma::dot(z, z);
            Rx = std::exp(-qx);
          }
          
          const double base = Rs * Rx;
          
          dres[ptheta - 1](a, b) = -base;                 // d/d alpha
          dres[0](a, b)          = off_scale * dRs_dphi * Rx; // d/d phi
          
          if (D_x > 0) {
            for (arma::uword l = 0; l < k; ++l) {
              const double zl = z(l);
              const double coeff = off_scale * Rs * Rx * (-2.0 * zl);
              const arma::uword col_offset = 1 + l * D_x;
              for (arma::uword p = 0; p < D_x; ++p) {
                dres[col_offset + p](a, b) = coeff * dx(p);
              }
            }
          }
        }
      }
    }
    
    return dres;
  }
  
  // ---- original gradients (matern != 3) ----
  if (theta.n_elem < 4) {
    Rcpp::stop("Correlationf_grad: expected theta length >= 4 for matern != 3.");
  }
  
  const double phi     = theta(0);
  const double sigmasq = theta(1);
  const double nu      = theta(2);
  const double alpha   = theta(3);
  
  std::vector<arma::mat> dres(4);
  for (auto &M : dres) M.set_size(ix.n_rows, iy.n_rows);
  
  if (matern == 1) {
    matern_grad_halfint_inplace(dres, coords, ix, iy, phi, nu, sigmasq, alpha, same);
  } else {
    Rcpp::stop("Correlationf_grad: unsupported gradients.");
  }
  
  return dres;
}

//[[Rcpp::export]]
arma::mat Correlationc(
    const arma::mat& coordsx,
    const arma::mat& coordsy,
    const arma::vec& theta,
    int matern,
    bool same){

  int nthreads = 1;
  //int bessel_ws_inc = 3;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(nthreads*MAT_NU_MAX, sizeof(double));
  
  if(same){
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    
    return Correlationf(coordsx, ix, ix, theta, bessel_ws, matern, same);
  } else {
    arma::mat coords = arma::join_vert(coordsx, coordsy);
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    arma::uvec iy = arma::regspace<arma::uvec>(coordsx.n_rows, coords.n_rows-1);
    
    return Correlationf(coords, ix, iy, theta, bessel_ws, matern, same);
  }
  
}
