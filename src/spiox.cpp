#include "spiox.h"

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart){
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tstart).count();
}

void SpIOX::update_running_means(arma::mat& E_A, const arma::mat& A, bool pr){
  // vi_it is the internal iteration counter
  if(pr){
    // simple moving average
    if(vi_it > vi_min_iter){
      E_A = E_A + (A - E_A) / (vi_it - vi_min_iter + 1.0);  
    }
  } else {
    double gamma = 0.65;
    double alpha = vi_it < vi_min_iter ? 1 : 1.0/pow(vi_it - vi_min_iter + 1.0, gamma); // weight for present
    E_A = (1-alpha) * E_A + alpha * A;
  }
}

void SpIOX::init_theta_adapt(){
  // adaptive metropolis
  theta_mcmc_counter = 0;
  which_theta_elem = arma::zeros<arma::uvec>(0);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  if(phi_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 0*oneuv);
  }
  if(sigmasq_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 1*oneuv);
  }
  if(nu_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 2*oneuv);
  }
  if(alpha_sampling){
    which_theta_elem = arma::join_vert(which_theta_elem, 3*oneuv);
  }
  
  int n_theta_par = q * which_theta_elem.n_elem;
  
  arma::mat bounds_all = arma::zeros(4, 2); // make bounds for all, then subset
  bounds_all.row(0) = arma::rowvec({.3, 200}); // phi
  bounds_all.row(1) = arma::rowvec({1e-6, 100}); // sigma
  if(matern){
    bounds_all.row(2) = arma::rowvec({1e-5, 2.1}); // nu  
  } else {
    // power exponential
    bounds_all.row(2) = arma::rowvec({1, 2}); // nu
  }
  
  //bounds_all.row(3) = arma::rowvec({1e-16, 100}); // tausq
  // 1-alpha is the proportion of variance explained by spatial component
  // alpha is is the proportion explained by nugget effect
  bounds_all.row(3) = arma::rowvec({1e-10, 1-1e-10}); // alpha
  bounds_all = bounds_all.rows(which_theta_elem);
  theta_unif_bounds = arma::zeros(0, 2);
  
  for(int j=0; j<q; j++){
    theta_unif_bounds = arma::join_vert(theta_unif_bounds, bounds_all);
  }
  
  arma::mat theta_metrop_sd = 0.05 * arma::eye(n_theta_par, n_theta_par);
  theta_adapt = RAMAdapt(n_theta_par, theta_metrop_sd, 0.24);
  theta_adapt_active = true;
  
  if(q > 2){
    // conditional update for more than 2 outcomes
    c_theta_unif_bounds = bounds_all;
    int c_theta_par = which_theta_elem.n_elem;
    arma::mat c_theta_metrop_sd = 0.05 * arma::eye(c_theta_par, c_theta_par);
    c_theta_adapt = std::vector<RAMAdapt>(q);
    for(int j=0; j<q; j++){
      c_theta_adapt[j] = RAMAdapt(c_theta_par, c_theta_metrop_sd, 0.24);
    }
    // ---  
  }
  
}

void SpIOX::compute_V(){ 
  // V is made of
  // B (in the response model only)
  // theta, and cluster assignments
  
  // it is used in 
  // Sigma
  // gibbs for theta
  
  // whiten the residuals from spatial dependence
  if(latent_model>0){
    V = W;
  } else {
    V = YXB;
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    V.col(j) = daggps.at(j).H_times_A(V.col(j), daggp_use_H);
  }
  
  bool do_VTV = vi & (latent_model>0);
  if(do_VTV){
    VTV = V.t() * V; 
  }
}

void SpIOX::update_B(){
  if(latent_model==0){
    arma::vec daggp_logdets = arma::zeros(q);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (unsigned int j = 0; j < q; ++j) {
      daggp_logdets(j)  = daggps.at(j).precision_logdeterminant;
      Ytilde.col(j)     = daggps.at(j).H_times_A(Y.col(j), daggp_use_H);
      HX.slice(j)       = daggps.at(j).H_times_A(X,        daggp_use_H);
    }
  
    arma::mat HX_mat(HX.memptr(), n, p * q, false, true);   // no-copy view of the cube
    arma::mat G = HX_mat.t() * HX_mat;                      // (qp x qp)
    
    arma::mat XtX(q * p, q * p);
    for (unsigned int a = 0; a < q; ++a) {
      for (unsigned int b = 0; b < q; ++b) {
        XtX.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1) =
          Q(a, b) *
          G.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1);
      }
    }
    
    arma::mat YS = Ytilde * Q;                          // n x q
    arma::vec XtY(q * p);
    for (unsigned int a = 0; a < q; ++a) {
      XtY.subvec(a * p, (a + 1) * p - 1) = HX.slice(a).t() * YS.col(a);
    }
    
    // Posterior precision and sample (if not vi)
    arma::mat post_precision = XtX;
    arma::vec vecB_Var = arma::vectorise(B_Var);
    post_precision.diag()   += 1.0 / vecB_Var;             
    
    arma::mat L        = arma::chol(post_precision, "lower");
    arma::mat pp_ichol = arma::inv(arma::trimatl(L));
    Beta_UQ            = pp_ichol.t() * pp_ichol;
    
    arma::mat randnormat = (vi ? 0.0 : 1.0) * arma::randn(p * q);
    arma::vec beta       = Beta_UQ * XtY + pp_ichol.t() * randnormat;
    B = arma::mat(beta.memptr(), p, q);
    
  } else {
    // update B via gibbs for the latent model
    // btw we could make this into a conjugate MN update rather than conj N
    Ytilde = Y - W;
    arma::mat mvnorm = arma::randn(p, q);
    for(int j=0; j<q; j++){
      arma::mat X_available = X.rows(avail_by_outcome(j));
      arma::mat XtX = X_available.t() * X_available;
      arma::mat post_precision = arma::diagmat(1.0/B_Var.col(j)) + XtX/Ddiag(j);
      arma::mat pp_ichol = arma::inv(arma::trimatl(arma::chol(post_precision, "lower")));
      arma::vec yj = Ytilde.col(j);
      arma::vec y_available = yj.rows(avail_by_outcome(j));
      arma::vec XtYtildej = arma::trans(X_available) * y_available;
      
      arma::vec B_mean = pp_ichol.t() * pp_ichol * XtYtildej/Ddiag(j);
      B.col(j) = B_mean + pp_ichol.t() * mvnorm.col(j);
      
    }
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::update_BW_asis(int& cg_iter, arma::mat& B, arma::mat& W, bool sampling){
  // Hold eta = XB + W constant; resample B then recover W = eta - XB.
  arma::mat eta = X * B + W;
  
  // whiten Yasis and X; HX[j] = H_j X 
  arma::mat Yasis(n, q);
  arma::vec daggp_logdets(q, arma::fill::zeros);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j = 0; j < q; ++j){
    Yasis.col(j)     = daggps.at(j).H_times_A(eta.col(j), daggp_use_H);
    HX.slice(j)      = daggps.at(j).H_times_A(X,          daggp_use_H);
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
  }
  
  //  Jacobi preconditioner 
  // P[j,j] diagonal at column c: Q(j,j) * ||HX[j].col(c)||^2 + 1/B_Var(c,j)
  arma::mat Mdiag_mat(p, q);
  for(unsigned int j = 0; j < q; ++j){
    arma::vec colsq(p);
    for(unsigned int c = 0; c < p; ++c){
      const arma::vec h = HX.slice(j).col(c);
      colsq(c) = arma::dot(h, h);
    }
    Mdiag_mat.col(j) = Q(j,j) * colsq + 1.0 / B_Var.col(j);
  }
  arma::vec Mdiag_vec = arma::vectorise(Mdiag_mat);
  
  const double diag_floor = 1e-12;
  for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
    if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
  
  auto apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
    z_out = r_in / Mdiag_vec;
  };
  
  
  // ----- Posterior precision multiply -----
  // (P b)[:,j] = HX[j]^T * (HX b * Q)[:,j] + (1/B_Var[:,j]) % b[:,j]
  arma::mat invBVar = 1.0 / B_Var;

  auto post_prec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat Bin (const_cast<double*>(x_in.memptr()), p, q, false, true);
    arma::mat Bout(y_out.memptr(),                     p, q, false, true);
    
    arma::mat HXb(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j) HXb.col(j) = HX.slice(j) * Bin.col(j);
    
    arma::mat HXbQ = HXb * Q;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j)
      Bout.col(j) = HX.slice(j).t() * HXbQ.col(j) + invBVar.col(j) % Bin.col(j);
  };
  
  arma::mat YasisQ = Yasis * Q;
  
  arma::mat Wsamp, Z2;
  if(sampling){
    Wsamp = arma::randn(n, q) * Si.t();
    Z2 = arma::randn(p, q);
  } else {
    Wsamp = arma::zeros(n, q);
    Z2 = arma::zeros(p, q);
}

  arma::mat invSqrtBVar = 1.0 / arma::sqrt(B_Var);
  arma::mat RHS_mat(p, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j = 0; j < q; ++j){
    RHS_mat.col(j) = HX.slice(j).t() * (YasisQ.col(j) + Wsamp.col(j))
    + invSqrtBVar.col(j) % Z2.col(j);
  }
  arma::vec post_meansample = arma::vectorise(RHS_mat);

  // ----- PCG solve -----
  arma::vec bbefore = arma::vectorise(B);
  arma::vec b = pcg_mf(post_prec_mv, apply_Minv, cg_iter, post_meansample,
                       bbefore, 1e-6, n, num_threads);

  B = arma::mat(b.memptr(), p, q);
  W = eta - X * B;
}

void SpIOX::gibbs_BW_block(int& cg_iter, PrecondChoice precond, bool sampling){
  // per-entry inverse noise variance (zero at missing)
  arma::mat invD_mat(n, q, arma::fill::zeros);
  arma::mat invSqrtD_mat(n, q, arma::fill::zeros);
  for(unsigned int j = 0; j < q; ++j){
    const double invDj  = 1.0 / Ddiag(j);
    const double invSDj = 1.0 / std::sqrt(Ddiag(j));
    for(unsigned int i = 0; i < n; ++i){
      if(!missing_mat(i, j)){
        invD_mat(i, j)     = invDj;
        invSqrtD_mat(i, j) = invSDj;
      }
    }
  }

  const arma::uword Nb = p * q;
  const arma::uword Nw = n * q;
  
  // ----- joint operator P_joint * (b ; w) -----
  // P_joint = blkdiag(Λ_B, Λ_W) + E^T D^{-1} E,  E = (A, I), A = I_q ⊗ X
  // Block layout in the vector: head = vec(B), tail = vec(W)
  auto post_prec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat Bin (const_cast<double*>(x_in.memptr()),       p, q, false, true);
    arma::mat Win (const_cast<double*>(x_in.memptr() + Nb),  n, q, false, true);
    arma::mat Bout(y_out.memptr(),                           p, q, false, true);
    arma::mat Wout(y_out.memptr() + Nb,                      n, q, false, true);
    
    // res = X*Bin + Win  (n × q),  res_sc = invD ⊙ res
    arma::mat res    = X * Bin + Win;
    arma::mat res_sc = invD_mat % res;
    
    // Λ_W * Win:  (Λ_W w).col(i) = H_i^T (Hw Q).col(i),  Hw.col(j) = H_j Win.col(j)
    arma::mat Hw(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j) Hw.col(j) = daggps[j].H * Win.col(j);
    arma::mat HwQ = Hw * Q;
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int i = 0; i < q; ++i)
      Wout.col(i) = daggps[i].H.t() * HwQ.col(i) + res_sc.col(i);
    
    // B-block:  out_B[:,j] = (1/B_Var[:,j]) ⊙ Bin[:,j] + X^T res_sc[:,j]
    arma::mat invBVar = 1.0 / B_Var;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(unsigned int j = 0; j < q; ++j)
      Bout.col(j) = invBVar.col(j) % Bin.col(j) + X.t() * res_sc.col(j);
  };
  
  // ----- preconditioner -----
  arma::vec Mdiag_vec;   // Jacobi
  arma::mat Sigma;       // Prior
  std::function<void(const arma::vec&, arma::vec&)> apply_Minv;
  
  if(precond == PRECOND_JACOBI){
    // W-half: Q(i,i) * (H_i^T H_i)(c,c) + invD_mat(c,i)
    arma::mat H_col_sq(n, q, arma::fill::zeros);
    for(unsigned int i = 0; i < q; ++i){
      daggps[i].H.sync();
      const double*      vals = daggps[i].H.values;
      const arma::uword* cols = daggps[i].H.col_ptrs;
      double* dst = H_col_sq.colptr(i);
      for(arma::uword c = 0; c < n; ++c){
        double s = 0.0;
        for(arma::uword k = cols[c]; k < cols[c+1]; ++k) s += vals[k]*vals[k];
        dst[c] = s;
      }
    }
    arma::mat Mdiag_W(n, q);
    for(unsigned int i = 0; i < q; ++i)
      Mdiag_W.col(i) = Q(i, i) * H_col_sq.col(i) + invD_mat.col(i);
    
    // B-half: 1/B_Var(c,j) + sum_i X(i,c)^2 invD(i,j)
    arma::mat Mdiag_B = (1.0 / B_Var) + arma::square(X).t() * invD_mat;
    
    Mdiag_vec.set_size(Nb + Nw);
    Mdiag_vec.head(Nb) = arma::vectorise(Mdiag_B);
    Mdiag_vec.tail(Nw) = arma::vectorise(Mdiag_W);
    
    const double diag_floor = 1e-12;
    for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
      if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
    
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      z_out = r_in / Mdiag_vec;
    };
      
      
  } else { // PRECOND_PPCG : block-diagonal prior preconditioner
    Sigma = arma::inv_sympd(Q);

    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      arma::mat RB(const_cast<double*>(r_in.memptr()),       p, q, false, true);
      arma::mat RW(const_cast<double*>(r_in.memptr() + Nb),  n, q, false, true);
      arma::mat ZB(z_out.memptr(),                           p, q, false, true);
      arma::mat ZW(z_out.memptr() + Nb,                      n, q, false, true);
      
      // B-half: Λ_B^{-1} = diag(B_Var)
      ZB = B_Var % RB;
      
      // W-half: Λ_W^{-1} via H solves and Σ mix (same as gibbs_w_block_PPCG)
      arma::mat Y(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for(unsigned int j = 0; j < q; ++j) Y.col(j) = daggps[j].Ht_solve_A(RW.col(j), true);
      arma::mat U = Y * Sigma;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for(unsigned int j = 0; j < q; ++j) ZW.col(j) = daggps[j].H_solve_A(U.col(j));
    };
  }
  
  arma::mat cW = invD_mat % Y;
  arma::mat cB = X.t() * cW;
  
  // prior noise on W (same Unorm trick as gibbs_w_block)
  arma::mat Unorm, xi_B_prior, Zlik_sc;
  if(sampling){
    Unorm = arma::randn(n, q) * Si.t();
    for(unsigned int j = 0; j < q; ++j) Unorm.col(j) = daggps[j].H.t() * Unorm.col(j);
    xi_B_prior = arma::randn(p, q) / arma::sqrt(B_Var);
    Zlik_sc = arma::randn(n, q) % invSqrtD_mat;
  } else {
    Unorm = arma::zeros(n, q);
    xi_B_prior = arma::zeros(p, q);
    Zlik_sc = arma::zeros(n, q);
  }
  
  // likelihood noise — SAME draw shared between B and W parts to match Λ_lik = E^T D^{-1} E
  arma::mat xi_B_lik = X.t() * Zlik_sc;
  
  arma::mat RHS_B = cB + xi_B_prior + xi_B_lik;
  arma::mat RHS_W = cW + Unorm     + Zlik_sc;
  
  arma::vec rhs(Nb + Nw);
  rhs.head(Nb) = arma::vectorise(RHS_B);
  rhs.tail(Nw) = arma::vectorise(RHS_W);

  // solve 
  arma::vec x0(Nb + Nw);
  x0.head(Nb) = arma::vectorise(B);
  x0.tail(Nw) = arma::vectorise(W);
  
  arma::vec sol = pcg_mf(post_prec_mv, apply_Minv, cg_iter, rhs, x0,
                         5*1e-5, n, num_threads);

  // unpack and keep YXB consistent 
  // YXB = Y - X B_new = (YXB_old + X B_old) - X B_new
  arma::mat B_old = B;
  B = arma::mat(sol.memptr(),       p, q);
  W = arma::mat(sol.memptr() + Nb,  n, q);
  YXB += X * (B_old - B);
}


void SpIOX::vi_Beta_UQ(){
  if(vi_it > vi_min_iter){
    arma::vec beta_v = arma::vectorise(B);
    
    delta_t = beta_v - beta_running_mean;
    beta_running_mean = beta_running_mean + 1.0/(1.0+vi_it-vi_min_iter) * delta_t;
    Beta_UQ = Beta_UQ + delta_t * arma::trans(beta_v - beta_running_mean);
  } else {
    beta_running_mean = arma::vectorise(B);
  }
}

bool SpIOX::upd_theta_metrop(){
  std::chrono::steady_clock::time_point tstart;
  std::chrono::steady_clock::time_point tend;
  int timed = 0;
  
  theta_adapt.count_proposal();
  
  arma::vec phisig_cur = arma::vectorise( theta.rows(which_theta_elem) );
  
  Rcpp::RNGScope scope;
  arma::vec U_update = arma::randn(phisig_cur.n_elem);
  
  arma::vec phisig_alt = par_huvtransf_back(par_huvtransf_fwd(
    phisig_cur, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  arma::mat theta_alt = theta;
  arma::mat phisig_alt_mat = arma::mat(phisig_alt.memptr(), which_theta_elem.n_elem, q);
  
  theta_alt.rows(which_theta_elem) = phisig_alt_mat; 
  
  if(!theta_alt.is_finite()){
    Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
  }
  
  // ---------------------
  // by default this runs when q=1 or q=2, no need for omp
  // create proposal daggp
  // this can run in parallel but update_theta already uses omp
  // do not run this in parallel, will be faster this way
  tstart = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<q; i++){
    daggps_alt[i].update_theta(theta_alt.col(i), daggp_use_H);
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "update dag proposal: " << timed << endl;
  // ----------------------
  // current density and proposal density
  tstart = std::chrono::steady_clock::now();
  
  arma::vec daggp_logdets = arma::zeros(q);
  arma::mat V_alt = V;
  arma::vec daggp_alt_logdets = arma::zeros(q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    arma::mat Target;
    if(latent_model>0){
      Target = W.col(j);
    } else {
      Target = YXB.col(j);
    }
    V_alt.col(j) = daggps_alt.at(j).H_times_A(Target, daggp_use_H);// * (Y.col(j) - X * B.col(j));
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
    daggp_alt_logdets(j) = daggps_alt.at(j).precision_logdeterminant;
  }
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "computing V: " << timed << endl;
  
  tstart = std::chrono::steady_clock::now();
  // current
  //arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  //arma::vec ytilde = arma::vectorise(V * Si);
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(V * Si, 2.0));
  
  // proposal
  //arma::mat Ytildemat_alt = arma::mat(vecYtilde_alt.memptr(), n, q, false, true);
  //arma::vec ytilde_alt = arma::vectorise(V_alt * Si);
  double prop_ldet = +0.5*arma::accu(daggp_alt_logdets);
  double prop_logdens = prop_ldet - 0.5*arma::accu(pow(V_alt * Si, 2.0));
  
  tend = std::chrono::steady_clock::now();
  timed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
  //Rcpp::Rcout << "computing VSi: " << timed << endl;
  
  // priors
  double logpriors = 0;
  for(unsigned int j=0; j<q; j++){
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      // nned to change this prior if alpha is a proportion which we expect ~0
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
  }
  
  // ------------------
  // make move
  double jacobian  = calc_jacobian(phisig_alt, phisig_cur, theta_unif_bounds);
  double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
  
  bool accepted = do_I_accept(logaccept);
  
  if(accepted){
    theta = theta_alt;
    std::swap(daggps, daggps_alt);
    std::swap(V, V_alt);
  } 
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
  }
  
  theta_mcmc_counter++;
  
  return accepted;
}

arma::uvec SpIOX::upd_theta_metrop_conditional(){
  
  using namespace std::chrono;
  auto t_start = steady_clock::now();
  long t_proposal = 0; // update dags with new theta proposals beforehand
  long t_mcmc  = 0; // sequential portion
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  arma::uvec accepteds = arma::zeros<arma::uvec>(q);
  
  // we can prepare the proposals beforehand, then only do the strictly sequential part
  
  Rcpp::RNGScope scope;
  arma::mat U_update = arma::randn(which_theta_elem.n_elem, q);
  arma::mat phisig_alt = arma::zeros(which_theta_elem.n_elem, q);
  arma::mat phisig_cur = arma::zeros(which_theta_elem.n_elem, q);
  arma::mat theta_alt = theta; // proposals
  
  for(int j=0; j<q; j++){
    c_theta_adapt[j].count_proposal();
    phisig_cur.col(j) = theta(which_theta_elem, oneuv*j);
    
    phisig_alt.col(j) = par_huvtransf_back(par_huvtransf_fwd(
      phisig_cur.col(j), c_theta_unif_bounds) + 
        c_theta_adapt[j].paramsd * U_update.col(j), c_theta_unif_bounds);
  
    // proposal for theta matrix
    //arma::mat theta_alt = theta;
    theta_alt(which_theta_elem, oneuv*j) = phisig_alt.col(j); 
    
    if(!theta_alt.is_finite()){
      Rcpp::stop("Some value of theta outside of MCMC search limits.\n");
    }
  }
  auto t0 = steady_clock::now();
  
  if(gridded){
    // gridded data -- 
#ifdef _OPENMP
#pragma omp parallel for num_threads(q) //***
#endif
    for(int j=0; j<q; j++){
      // create proposal daggp // 1 thread here
      daggps_alt[j].update_theta(theta_alt.col(j), daggp_use_H);
    }
  } else {
    for(int j=0; j<q; j++){
      // create proposal daggp // omp inside here -- its slightly faster
      daggps_alt[j].update_theta(theta_alt.col(j), daggp_use_H);
    }
  }
  

  
  auto t1 = steady_clock::now();
  t_proposal += duration_cast<microseconds>(t1 - t0).count();
  
  // sequential
  for(int j=0; j<q; j++){
    // conditional density of Y_j | Y_-j (or W depending on target)
    arma::mat V_alt = V;
    
    if(latent_model>0){
      V_alt.col(j) = daggps_alt.at(j).H_times_A(W.col(j), daggp_use_H);// * (Y.col(j) - X * B.col(j));
    } else {
      V_alt.col(j) = daggps_alt.at(j).H_times_A(YXB.col(j), daggp_use_H);// * (Y.col(j) - X * B.col(j));
    }
    
    double c_daggp_logdet = daggps.at(j).precision_logdeterminant;
    double c_daggp_alt_logdet = daggps_alt.at(j).precision_logdeterminant;
    
    arma::vec Vjc = arma::zeros(n);
    arma::vec Vjc_alt = arma::zeros(n);
    for(int jc=0; jc<q; jc++){
      Vjc += Q(j, jc)/Q(j,j) * V.col(jc);
      Vjc_alt += Q(j, jc)/Q(j,j) * V_alt.col(jc);
    }
    
    double core_alt = arma::accu(pow(Vjc_alt, 2.0)); 
    double core = arma::accu(pow(Vjc, 2.0)); 
    double prop_logdens = 0.5 * c_daggp_alt_logdet - Q(j,j)/2.0 * core_alt;
    double curr_logdens = 0.5 * c_daggp_logdet - Q(j,j)/2.0 * core;
    
    // priors
    double logpriors = 0;
    if(sigmasq_sampling){
      logpriors += invgamma_logdens(theta_alt(1,j), 2, 1) - invgamma_logdens(theta(1,j), 2, 1);
    }
    if(alpha_sampling){
      //logpriors += expon_logdens(theta_alt(3,j), 25) - expon_logdens(theta(3,j), 25);
    }
    
    // ------------------
    // make move
    double jacobian  = calc_jacobian(phisig_alt.col(j), phisig_cur.col(j), c_theta_unif_bounds);
    double logaccept = prop_logdens - curr_logdens + jacobian + logpriors;
    
    accepteds(j) = do_I_accept(logaccept);
    
    if(accepteds(j)){
      theta.col(j) = theta_alt.col(j);
      std::swap(daggps.at(j), daggps_alt.at(j));
      //std::swap(V, V_alt);
      V.col(j) = V_alt.col(j);
    } 
    
    c_theta_adapt[j].update_ratios();
    
    if(theta_adapt_active){
      c_theta_adapt[j].adapt(U_update.col(j), exp(logaccept), theta_mcmc_counter); 
    }
    
    theta_mcmc_counter++;
  }
  
  auto t2 = steady_clock::now();
  t_mcmc += duration_cast<microseconds>(t2 - t1).count();
  
  // --- PRINT RESULTS ---
  //Rcpp::Rcout << "--- Conditional Update Timing (microseconds) ---" << std::endl;
  //Rcpp::Rcout << "Proposal build:   " << t_proposal << std::endl;
  //Rcpp::Rcout << "MCMC seq:      " << t_mcmc << std::endl;
  
  return accepteds;
}

void SpIOX::cache_blanket_comps(const arma::uvec& theta_changed){
  // all these matrix operations only need to be performed when theta changes
  // otherwise, we can just cache what's needed and reuse
  // this impacts the latent model, single site sampler
  // and imputation of missing data in the response model.
  int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) //***
#endif
  for(int ix=0; ix<nfill; ix++){
    int i = latent_model == 2 ? ix : rows_with_missing(ix);
    
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
  
    // initialization
    Rw_no_Q(ix) = arma::zeros(q, q); 
    Pblanket_no_Q(ix) = arma::zeros(q, q*mbsize);

    for(int r=0; r<q; r++){
      for(int s=0; s<=r; s++){
        Rw_no_Q(ix)(r, s) = 
          arma::accu( daggps[r].H.col(i) % 
          daggps[s].H.col(i) );
        if(s!=r){
          Rw_no_Q(ix)(s, r) = Rw_no_Q(ix)(r, s);
        }
      }
    }
    
    arma::sp_mat H_i_mat(n, q);
    for (int r = 0; r < q; ++r) {
      H_i_mat.col(r) = daggps[r].H.col(i);
    }
    arma::sp_mat Hitt = H_i_mat.t();
    
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      arma::sp_mat Hblanket = daggps[s].H.cols(mblanket);
      arma::mat result(Hitt * Hblanket);
      Pblanket_no_Q(ix).cols(startcol, endcol) = result;
    }
  }
}

void SpIOX::w_sequential_singlesite(const arma::uvec& theta_changed){
  double ms_if_cache = 0;
  double ms_omp_for = 0;
  double ms_sample = 0;
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  
  arma::cube Postcov = arma::zeros(q, q, n);
  arma::mat randcomp = arma::randn(q, n);
  //arma::mat mvnorm = arma::randn(q, n);
  
  // V = whitened Y-XB or W
  if(arma::any(theta_changed != 0)){
    // perform this update if theta has changed and we need to recompute the
    // GP-related matrices that depend on it
    cache_blanket_comps(theta_changed);
  }
  
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    Rw(i) = Q % Rw_no_Q(i);
    
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    arma::mat Di_obs = arma::zeros(q,q);
    
    for(int j = 0; j < q; j++) {
      int startcol = j * mbsize;
      int endcol = (j + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(j)) * Pblanket_no_Q(i).cols(startcol, endcol);
      if(!missing_mat(i,j)){
        Di_obs(j,j) = 1.0/Ddiag(j);
      }
    }
    
    Hw(i) = - Pblanket;
    arma::mat invcholP = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di_obs, "lower")));
    Postcov.slice(i) = invcholP.t() * invcholP;
    randcomp.col(i) = invcholP.t() * randcomp.col(i);
  }

  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int c=0; c < daggps[0].colors.n_elem; c++){
    arma::uvec nodes_in_color = daggps[0].colors(c);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int ix=0; ix < nodes_in_color.n_elem; ix++){
      int i = nodes_in_color(ix);
      //for(int i=0; i<n; i++){
      // data contributions may be null if data missing
      arma::vec Di_YXB = arma::zeros(q);
      for(int j=0; j<q; j++){
        if(!missing_mat(i,j)){
          Di_YXB(j) = YXB(i,j)/Ddiag(j);
        }
      }
      
      arma::uvec mblanket = daggps[0].mblanket(i);
    
      // sample
      arma::vec W_mean = Postcov.slice(i) * ( Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di_YXB );
      W.row(i) = arma::trans(  W_mean + randcomp.col(i) );
      W_RB.row(i) = W_mean.t();
      
      if(W.has_nan()){
        Rcpp::stop("Found nan in W.\n");
      }
    }
  }


}

/*
void SpIOX::w_sequential_singlesite(const arma::uvec& theta_changed, bool vi=false){
  // stuff to be moved to SpIOX class for latent model
  arma::mat Di = arma::diagmat(1/Ddiag);
  
  // precompute stuff in parallel so we can do fast sequential sampling after
  arma::mat mvnorm = arma::randn(q, n);
  
  arma::field<arma::mat> Hw(n);
  arma::field<arma::mat> Rw(n);
  arma::field<arma::mat> Ctchol(n);
  
  // V = whitened Y-XB or W
  
#ifdef _OPENMP
//#pragma omp parallel for num_threads(num_threads)
#endif
  for(int i=0; i<n; i++){
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    
    Rw(i) = arma::zeros(q, q); 
    arma::mat Pblanket(q, q*mbsize);
    for(int r=0; r<q; r++){
      for(int s=0; s<q; s++){
        Rw(i)(r, s) = Q(r,s) * 
          arma::accu( daggps[r].H.col(i) % 
          daggps[s].H.col(i) );
        
        int startcol = s * mbsize;
        int endcol = (s + 1) * mbsize - 1;
        for(int j = 0; j < mblanket.n_elem; j++) {
          int col_idx = mblanket(j);
          Pblanket(r, startcol + j) = Q(r, s) *
            arma::accu(daggps[r].H.col(i) %
            daggps[s].H.col(col_idx));
        }
      }
    }
    
    Hw(i) = - Pblanket;
    Ctchol(i) = arma::inv(arma::trimatl(arma::chol(Rw(i) + Di, "lower")));
  }
  
  // visit every location and sample from latent effects 
  // conditional on data and markov blanket
  for(int i=0; i<n; i++){
    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::vec meancomp = Hw(i) * arma::vectorise( W.rows(mblanket) ) + Di * arma::trans(YXB.row(i)); 
    
    W.row(i) = arma::trans( Ctchol(i).t() * (Ctchol(i) * meancomp + 0*mvnorm.col(i) ));
  }
  
}
*/

void SpIOX::gibbs_w_sequential_byoutcome(){
  
  std::vector<arma::sp_mat> prior_precs(q);
  
  arma::mat urands = arma::randn(n, q);
  arma::mat vrands = arma::randn(n, q);
  
  arma::mat HDs = arma::zeros(n, q);
  arma::uvec r1q = arma::regspace<arma::uvec>(0,q-1);
  
  YXB.elem(find(missing_mat)).zeros();
  
  for(int j=0; j<q; j++){
    // build per-location diagonal contribution for this column j
    arma::vec invD(n, arma::fill::zeros);
    arma::vec invSqrtD(n, arma::fill::zeros);
    for(int i = 0; i < n; i++){
      if(!missing_mat(i,j)){
        invD(i)      = 1.0 / Ddiag(j);
        invSqrtD(i)  = 1.0 / std::sqrt(Ddiag(j));
      }
    }
    
    // prior precision
    arma::sp_mat post_prec = Q(j,j) * daggps[j].Ci;
    
    // posterior precision: add only where observed
    post_prec.diag() += invD;
    
    // data/noise contribution: zeroed out where missing
    // (elementwise multiply by invD / invSqrtD vectors)
    HDs.col(j) =
      YXB.col(j) % invD
    + urands.col(j) % invSqrtD
    + std::sqrt(Q(j,j)) * daggps[j].H.t() * vrands.col(j);
    
    arma::vec x = arma::zeros(n);
    arma::uvec notj = arma::find(r1q != j);
    arma::uvec jx = arma::zeros<arma::uvec>(1) + j;
    
    // V is whitened W as we want
    arma::vec Mi_m_prior = - daggps[j].H.t() * V.cols(notj) * Q.submat(notj, jx);
    arma::vec rhs = Mi_m_prior + HDs.col(j);
    
    W.col(j) = pcg_diag_solve(post_prec, rhs, W.col(j), 1e-5, 500, 1e-14, num_threads);
    V.col(j) = daggps[j].H_times_A(W.col(j)); // keep here
  }
  
}

void SpIOX::gibbs_w_block(int& cg_iter, PrecondChoice precond){
  // noise sample with covariance A_prior
  arma::mat Unorm = arma::randn(n, q) * Si.t();
  for(int i = 0; i < q; ++i)
    Unorm.col(i) = daggps[i].H.t() * Unorm.col(i);
  
  // per-entry inverse noise variance (zero at missing)
  arma::mat invD_mat(n, q, arma::fill::zeros);
  arma::mat invSqrtD_mat(n, q, arma::fill::zeros);
  for(int j = 0; j < q; ++j){
    const double invDj  = 1.0 / Ddiag(j);
    const double invSDj = 1.0 / std::sqrt(Ddiag(j));
    for(int i = 0; i < n; ++i){
      if(!missing_mat(i, j)){
        invD_mat(i, j)     = invDj;
        invSqrtD_mat(i, j) = invSDj;
      }
    }
  }
  
  // A * v  (shared operator) 
  auto post_prec_mv = [&](const arma::vec& x_in, arma::vec& y_out){
    arma::mat X(const_cast<double*>(x_in.memptr()), n, q, false, true);
    arma::mat Y(y_out.memptr(),                     n, q, false, true);
    
    arma::mat Hx(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int j = 0; j < q; ++j) Hx.col(j) = daggps[j].H * X.col(j);
    
    arma::mat HxQ = Hx * Q;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int i = 0; i < q; ++i)
      Y.col(i) = daggps[i].H.t() * HxQ.col(i) + invD_mat.col(i) % X.col(i);
  };
  
  // preconditioner 
  arma::vec Mdiag_vec;   // Jacobi
  std::function<void(const arma::vec&, arma::vec&)> apply_Minv;
  
  if(precond == PRECOND_JACOBI){
    // Per-outcome column-squared-norms of H_i, used to build Mdiag.
    // (diag(H_i^T H_i)(c) = sum_k H_i(k,c)^2)
    arma::mat H_col_sq(n, q, arma::fill::zeros);
    for(int i = 0; i < q; ++i){
      daggps[i].H.sync();
      const double*      vals = daggps[i].H.values;
      const arma::uword* cols = daggps[i].H.col_ptrs;
      double* dst = H_col_sq.colptr(i);
      for(arma::uword c = 0; c < n; ++c){
        double s = 0.0;
        for(arma::uword k = cols[c]; k < cols[c+1]; ++k) s += vals[k]*vals[k];
        dst[c] = s;
      }
    }
    
    arma::mat Mdiag_mat(n, q);
    for(int i = 0; i < q; ++i)
      Mdiag_mat.col(i) = Q(i,i) * H_col_sq.col(i) + invD_mat.col(i);
    Mdiag_vec = arma::vectorise(Mdiag_mat);
    
    // Floor once, up front, so the closure is branchless.
    const double diag_floor = 1e-12;
    for(arma::uword i = 0; i < Mdiag_vec.n_elem; ++i)
      if(!(Mdiag_vec(i) > diag_floor)) Mdiag_vec(i) = diag_floor;
      
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      z_out = r_in / Mdiag_vec;
    };
      
  } else { 
    apply_Minv = [&](const arma::vec& r_in, arma::vec& z_out){
      arma::mat R(const_cast<double*>(r_in.memptr()), n, q, false, true);
      arma::mat Z(z_out.memptr(),                     n, q, false, true);
      
      arma::mat Y(n, q);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for(int j = 0; j < q; ++j) Y.col(j) = daggps[j].Ht_solve_A(R.col(j), true);
      
      arma::mat U = Y * Sigma;
      
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
      for(int j = 0; j < q; ++j) Z.col(j) = daggps[j].H_solve_A(U.col(j));
    };
  }
  
  // ----- RHS -----
  arma::vec post_Cmean      = arma::vectorise(YXB % invD_mat);
  arma::vec vnorm           = arma::vectorise(arma::randn(n, q) % invSqrtD_mat);
  arma::vec post_meansample = post_Cmean + arma::vectorise(Unorm) + vnorm;
  
  arma::vec wbefore = arma::vectorise(W);
  arma::vec w = pcg_mf(post_prec_mv, apply_Minv, cg_iter, post_meansample,
                       wbefore, 1e-5, n, num_threads);
  
  W = arma::mat(w.memptr(), n, q);
}

void SpIOX::update_Ddiag_gibbs(){
  arma::mat E = YXB - W;

  // priors for tau_sq
  double a = 2; // 1e-5;
  double b = 1; // 1e-5;
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    double navail = .0 + avail_by_outcome(j).n_elem;

    // MCMC - sample with inverse gamma for each tau_sq
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    double ssq = arma::accu(arma::square(ej_sub));
    Ddiag(j) = 1.0/R::rgamma(navail/2 + a, 1.0/(b + 0.5 * ssq));
  
  }
}

void SpIOX::update_Ddiag_vi(){
  arma::mat E = YXB - W;
  
  // priors for tau_sq
  double a = 2;//1e-5;
  double b = 1;//1e-5;
  
  Ddiag_UQ = arma::zeros(q);
  ETE = arma::zeros(q, q);

  for(int j = 0; j < q; j++){
    arma::uvec ix = avail_by_outcome(j);
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    if(ej_sub.n_elem > 0){
      ej_sub -= arma::mean(ej_sub);
      ETE(j, j) = arma::accu(arma::square(ej_sub)); 
    }
  }
  update_running_means(ETE_ma, ETE);
  
  // Updating each tau_sq
  for(int j=0; j<q; j++){
    double navail = -1.0 + avail_by_outcome(j).n_elem;
    Ddiag_UQ(j) = ETE_ma(j,j);
    Ddiag(j) = (b + 0.5 * Ddiag_UQ(j)) / (navail/2 + a - 1);
  }
}

void SpIOX::W_centering(){
  if(intercept != -1){
    // we have an intercept. move the mean of W to it
    arma::rowvec w_means = arma::mean(W, 0);
    W.each_row() -= w_means;
    B.row(intercept) += w_means;
    YXB.each_row() -= w_means;
  }
}

void SpIOX::update_Sigma_iwishart(){
  arma::mat Smean = V.t() * V + arma::eye(V.n_cols, V.n_cols);
  arma::mat Q_mean_post;
  
  try { 
    Q_mean_post = arma::inv_sympd(Smean);
  } catch (...) {
    Rcpp::Rcout << Smean << std::endl;
    // Rcpp::Rcout << theta << std::endl; // uncomment if theta is in scope
    Rcpp::stop("Error in inv_sympd within sample_Sigma_iwishart \n");
  }
  
  double df_post = n + (V.n_cols);
  
  Q = arma::wishrnd(Q_mean_post, df_post);
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  Sigma = S.t() * S;
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}

void SpIOX::update_Sigma_vi(){
  if(latent_model > 0){
    update_running_means(VTV_ma, VTV);
    Sigma_UQ = arma::eye(q,q) + VTV_ma;
  } else {
    arma::mat HX_mat(HX.memptr(), n, p * q, false, true);   // no-copy view of the cube
    arma::mat G = HX_mat.t() * HX_mat;                       // (pq x pq)
    
    arma::mat E2(q, q, arma::fill::zeros);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int a = 0; a < (int)q; ++a) {
      for (unsigned int b = a; b < q; ++b) {                 // upper triangle only
        E2(a, b) = arma::accu(
          Beta_UQ.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1) %
            G.submat(a * p, b * p, (a + 1) * p - 1, (b + 1) * p - 1)
        );
      }
    }
    
    Sigma_UQ = arma::eye(q, q) + V.t() * V + arma::symmatu(E2);
  }
  
  double df_post = q + n;
  Sigma = Sigma_UQ / (df_post - q - 1);  
  Q = arma::inv_sympd(Sigma);
  Si = arma::chol(Q, "lower");
  S = arma::inv(arma::trimatl(Si));
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}

void SpIOX::sample_Y_misaligned(const arma::uvec& theta_changed){
  // precompute stuff in parallel so we can do fast sequential sampling after
  int nfill = rows_with_missing.n_elem;
  arma::mat mvnorm = arma::randn(q, nfill);
  
  // for numerical stability --
  arma::vec col_sd(q);
  arma::vec col_mean(q); // residual mean, should be ~0 but just in case
  
  for(int j = 0; j < q; j++){
    arma::uvec av = avail_by_outcome(j);
    // residuals for available observations
    arma::vec resid_j = Y(av, arma::uvec({(unsigned)j})) - 
      X.rows(av) * B.col(j);
    col_sd(j) = arma::stddev(resid_j);
    col_mean(j) = arma::mean(resid_j);
    if(col_sd(j) < 1e-10) col_sd(j) = 1.0;
  }
  // 

  arma::field<arma::mat> Hw(nfill);
  arma::field<arma::mat> Rw(nfill);
  arma::field<arma::mat> invcholP(nfill);
  
  
  if(arma::any(theta_changed != 0)){
    
    // perform this update if theta has changed and we need to recompute the
    // GP-related matrices that depend on it
    cache_blanket_comps(theta_changed);
  }
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(int ix=0; ix<nfill; ix++){
    // this row has missing data
    int i = rows_with_missing(ix);
    Rw(ix) = Q % Rw_no_Q(ix);
    // assume all the same dag otherwise we go cray
    arma::uvec mblanket = daggps[0].mblanket(i);
    int mbsize = mblanket.n_elem;
    arma::mat Pblanket = arma::zeros(q, q*mbsize);
    
    for (int s = 0; s < q; ++s) {
      int startcol = s * mbsize;
      int endcol = (s + 1) * mbsize - 1;
      Pblanket.cols(startcol, endcol) = arma::diagmat(Q.col(s)) * Pblanket_no_Q(ix).cols(startcol, endcol);
    }
    Hw(ix) = - Pblanket;
    invcholP(ix) = arma::inv(
      arma::trimatl(arma::chol(Rw(ix), "lower")));
  }
  
  // visit every location with missing data and fill 
  // conditional on what's available and markov blanket
  for(int ix=0; ix<nfill; ix++){
    arma::vec rnormq = mvnorm.col(ix); // preloaded sample size q
    // this row has missing
    int i = rows_with_missing(ix);
    arma::uvec mblanket = daggps[0].mblanket(i);
    arma::mat YXB_others = YXB.rows(mblanket);
    
    arma::uvec which_missing = arma::find(missing_mat.row(i) == 1);
    if(which_missing.n_elem == q){
      // everything missing!
      arma::mat meancomp = invcholP(ix) * Hw(ix) * arma::vectorise( YXB_others );
      Y.row(i) = arma::trans(invcholP(ix).t() * (meancomp + rnormq)) + X.row(i) * B;
    } else {
      // some data available at this location
      arma::mat joint_cov = invcholP(ix).t() * invcholP(ix);
      arma::vec joint_mean = joint_cov * Hw(ix) * arma::vectorise( YXB_others );
      
      arma::uvec which_availab = arma::find(missing_mat.row(i) == 0);
      
      arma::mat Ckk = joint_cov(which_availab, which_availab);
      arma::mat Ckx = joint_cov(which_availab, which_missing);
      arma::mat Cxx = joint_cov(which_missing, which_missing);
      arma::mat HmatT = arma::solve(Ckk, Ckx);
      
      arma::mat cholRmat = arma::chol(arma::symmatu(Cxx - Ckx.t() * HmatT), "lower");
      
      arma::vec Yall = arma::trans(YXB.row(i));
      Yall(which_missing) = joint_mean(which_missing) + 
        HmatT.t() * (Yall(which_availab) - joint_mean(which_availab)) + 
        cholRmat * rnormq(which_missing);
      
      Y.row(i) = Yall.t() + X.row(i) * B;
    }
    
    // for numerical stability 
    for(unsigned jj = 0; jj < which_missing.n_elem; jj++){
      unsigned j = which_missing(jj);
      // fitted value for this observation and outcome
      double fitted_ij = arma::dot(X.row(i), B.col(j));
      double lo = fitted_ij + col_mean(j) - 3.0 * col_sd(j);
      double hi = fitted_ij + col_mean(j) + 3.0 * col_sd(j);
      double val = Y(i, j);
      Y(i, j) = val < lo ? lo : (val > hi ? hi : val);
    }
    
  }
  
  YXB = Y - X*B;
  
}

void SpIOX::update_BWSigma_px(){
  arma::mat SA = arma::zeros(q,q);
  arma::mat randnormmat = arma::randn(p+q, q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    arma::vec yj = Y.col(j);
    arma::mat Zj = daggps[j].H_solve_A(V, daggp_use_H);
    
    yj = yj.rows(ix);
    arma::mat ZZ = arma::join_horiz(X.rows(ix), Zj.rows(ix));
    
    arma::mat prior_precision = arma::zeros(p+q, p+q);
    prior_precision.submat(0, 0, p-1, p-1) = arma::diagmat(1.0/B_Var.col(j));
    
    arma::mat post_precision = prior_precision + ZZ.t() * ZZ / Ddiag(j);
    arma::mat cholP = arma::chol(arma::symmatu(post_precision), "lower");
    
    arma::mat cholV = arma::inv(arma::trimatl(cholP));
    arma::vec BAj = cholV.t() * ( cholV * ZZ.t() * yj / Ddiag(j) + randnormmat.col(j));
    
    B.col(j) = BAj.head_rows(p);
    SA.col(j) = BAj.tail_rows(q);
    
    W.col(j) = Zj * SA.col(j);
    YXB.col(j) = Y.col(j) - X * B.col(j);
  }
  
  Sigma = SA.t() * Sigma * SA;
  S = arma::chol(Sigma, "upper");
  Si = arma::inv(arma::trimatu(S));
  Q = Si * Si.t();
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}

void SpIOX::response_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq){
  
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  arma::uvec theta_has_changed = arma::zeros<arma::uvec>(q);
  if(update_theta){
    if(q>2){
      theta_has_changed = upd_theta_metrop_conditional(); 
    } else {
      bool block_changed = upd_theta_metrop();
      theta_has_changed += block_changed;
    }
  }
  timings(4) += time_count(tstart);
  
  if(sample_beta){
    // sample B 
    tstart = std::chrono::steady_clock::now();
    update_B();
    timings(0) += time_count(tstart);  
    
    // need to recompute V only when V = Y-XB (response model)

    tstart = std::chrono::steady_clock::now();
    compute_V();
    timings(1) += time_count(tstart);
  
  }

  
  tstart = std::chrono::steady_clock::now();
  // response model -- do we have missing data? if so, impute
  if(Y_needs_filling){
    // redo_cache_blanket runs if update_theta=true
    sample_Y_misaligned(theta_has_changed);
    compute_V();
  }
  timings(5) += time_count(tstart);

  
  if(sample_sigma > 0){
    tstart = std::chrono::steady_clock::now();
    update_Sigma_iwishart();
    timings(2) += time_count(tstart); 
  }
  
}

void SpIOX::latent_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq){
  
  //Rcpp::Rcout << "theta " << endl;
  // update atoms for theta
  tstart = std::chrono::steady_clock::now();
  arma::uvec theta_has_changed = arma::zeros<arma::uvec>(q);
  if(update_theta){
    if(q>2){
      theta_has_changed = upd_theta_metrop_conditional();
    } else {
      bool block_changed = upd_theta_metrop();
      theta_has_changed += block_changed;
    }
  }
  timings(4) += time_count(tstart);
  
  if(latent_model == 1){
    // sample Beta and W as a block
    int cg_iter = 0;
    
    if(precond_choice == PRECOND_PROBE){
      const PrecondChoice use = (probe_count % 2 == 0) ? PRECOND_JACOBI : PRECOND_PPCG;
      gibbs_BW_block(cg_iter, use);
      
      if(use == PRECOND_JACOBI){
        cost_jacobi += static_cast<double>(cg_iter);
        ++n_probe_jac;
      } else {
        cost_ppcg += static_cast<double>(cg_iter) * ppcg_iter_cost;
        ++n_probe_ppcg;
      }
      ++probe_count;
      
      if(probe_count >= probe_max && n_probe_jac > 0 && n_probe_ppcg > 0){
        const double avg_jac  = cost_jacobi / n_probe_jac;
        const double avg_ppcg = cost_ppcg   / n_probe_ppcg;
        precond_choice = (avg_jac <= avg_ppcg) ? PRECOND_JACOBI : PRECOND_PPCG;
        //Rcpp::Rcout << "[gibbs_BW_block] locking in "
        //            << (precond_choice == PRECOND_JACOBI ? "Jacobi" : "Prior")
        //            << " (avg cost: Jacobi=" << avg_jac
        //            << ", Prior=" << avg_ppcg << ")\n";
      }
    } else {
      gibbs_BW_block(cg_iter, precond_choice);
    }
  } else {
    //Rcpp::Rcout << "B " << endl;
    if(sample_beta){
      // sample B 
      tstart = std::chrono::steady_clock::now();
      update_B(); 
      timings(0) += time_count(tstart);  
    }
    
    if(latent_model == 4){
      // future
      int pg_iter = 0;
      //gibbs_w_block_woodbury(pg_iter, PRECOND_JACOBI);
    }
    
    if(latent_model == 2){
      // redo_cache_blanket runs if update_theta=true
      w_sequential_singlesite(theta_has_changed); 
    }
    if(latent_model == 3){
      gibbs_w_sequential_byoutcome();
    }
    timings(5) += time_count(tstart);
    
    //Rcpp::Rcout << "B asis " << endl;
    if(sample_beta){
      int cg_iter=0;
      update_BW_asis(cg_iter, B, W, true); // do sample
      YXB = Y - X*B;
    }
  }

  compute_V(); // keep 
  
  if(sample_sigma){
    //Rcpp::Rcout << "Sigma centered " << endl;
    tstart = std::chrono::steady_clock::now();
    update_Sigma_iwishart();
    
    //Rcpp::Rcout << "Sigma PX " << endl;
    // PX W~Sigma
    update_BWSigma_px();
    timings(2) += time_count(tstart); 
  }

  //Rcpp::Rcout << "W centering " << endl;
  W_centering(); // move to intercept if we have one
  compute_V();
  
  //Rcpp::Rcout << "Tausq " << endl;
  if(sample_tausq){
    update_Ddiag_gibbs();
  }
  
}

void SpIOX::response_vi(){
  Rcpp::Rcout << "B\n";
  update_B();
  Rcpp::Rcout << "V\n";
  // V = whitened Y-XB or W
  compute_V();
  Rcpp::Rcout << "S\n";
  update_Sigma_vi();
}

void SpIOX::latent_vi(){
  auto t0 = std::chrono::high_resolution_clock::now();
  auto t_prev = t0;
  auto checkpoint = [&](const char* label){
    auto t_now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    //Rcpp::Rcout << "[latent_vi] " << label << ": " << ms << " ms\n";
    t_prev = t_now;
  };
  
  int cg_iter = 0;
  if(precond_choice == PRECOND_PROBE){
    // Alternate Jacobi / PPCG during the probe phase.
    const PrecondChoice use = (probe_count % 2 == 0) ? PRECOND_JACOBI : PRECOND_PPCG;
    gibbs_BW_block(cg_iter, use); // gibbs_w_block
    
    if(use == PRECOND_JACOBI){
      cost_jacobi += static_cast<double>(cg_iter);   // unit cost per iter
      ++n_probe_jac;
    } else {
      cost_ppcg += static_cast<double>(cg_iter) * ppcg_iter_cost;
      ++n_probe_ppcg;
    }
    ++probe_count;
    
    // Lock in the winner once both have been measured enough.
    if(probe_count >= probe_max && n_probe_jac > 0 && n_probe_ppcg > 0){
      const double avg_jac  = cost_jacobi / n_probe_jac;
      const double avg_ppcg = cost_ppcg   / n_probe_ppcg;
      precond_choice = (avg_jac <= avg_ppcg) ? PRECOND_JACOBI : PRECOND_PPCG;
      
      //Rcpp::Rcout << "[vi_w_block] locking in "
      //            << (precond_choice == PRECOND_JACOBI ? "Jacobi" : "PPCG")
      //            << " (avg weighted cost: Jacobi=" << avg_jac
      //            << ", PPCG=" << avg_ppcg << ")\n";
    }
  } else {
    gibbs_BW_block(cg_iter, precond_choice); //gibbs_w_block
  }
  checkpoint("gibbs_BW_block");
  
  W_centering();
  checkpoint("W_centering #1");
  
  // working V with sampled B and W
  compute_V();
  checkpoint("compute_V #1");
  
  // PX
  update_BWSigma_px();
  checkpoint("update_BWSigma_px");
  
  W_centering();
  checkpoint("W_centering #2");
  
  compute_V();
  checkpoint("compute_V #2");
  
  // Update E(Sigma) using MC approx of E(quad forms)
  update_Sigma_vi();
  checkpoint("update_Sigma_vi");
  
  // Update E(Ddiag) using MC approx of E(quad forms)
  update_Ddiag_vi(); 
  checkpoint("update_Ddiag_vi");
  
  // compute covariance in B iteratively
  vi_Beta_UQ();
  checkpoint("vi_Beta_UQ");
  
  // update running means for E(B) and E(W)
  update_running_means(E_B, B);
  update_running_means(E_W, W, false); //**
  checkpoint("update_running_means");
  
  double total_ms = std::chrono::duration<double, std::milli>(
    std::chrono::high_resolution_clock::now() - t0).count();
  //Rcpp::Rcout << "[latent_vi] TOTAL: " << total_ms << " ms (cg_iter=" 
  //            << cg_iter << ")\n";
  
  vi_it ++;
}


void SpIOX::map(){
  arma::mat Ytilde = Y;
  
  Xtilde = arma::zeros(n*q, p*q);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for(unsigned int j=0; j<q; j++){
    Ytilde.col(j) = daggps.at(j).H_times_A(Y.col(j), daggp_use_H); // whitening
    arma::mat HX = daggps.at(j).H_times_A(X, daggp_use_H);         // whitened X
    for(unsigned int i=0; i<q; i++){
      Xtilde.submat(i * n,       j * p,
                    (i+1) * n-1, (j+1) * p - 1) = Si(j,i) * HX;
    }
  }
  
  arma::vec ytilde = arma::vectorise(Ytilde * Si); // vec(Y^* Σ^{-1})
  
  arma::vec vecB_Var = arma::vectorise(B_Var);
  arma::mat prior_prec = arma::diagmat(1.0 / vecB_Var);
  
  arma::mat post_precision = prior_prec + Xtilde.t() * Xtilde;
  arma::vec mu_b = arma::solve(post_precision, Xtilde.t() * ytilde);
  
  // store the mean and optionally the covariance (if needed for ELBO or sampling)
  B = arma::mat(mu_b.memptr(), p, q);  // reshape to matrix for consistency
  
  Beta_UQ = arma::inv_sympd(post_precision);
  
  // V = whitened Y-XB or W
  compute_V();
  
  // trace terms
  arma::mat E2 = arma::zeros(q, q);
  for (unsigned int i = 0; i < q; ++i){
    arma::mat Xi = Xtilde.rows(i * n, (i+1) * n - 1);
    for (unsigned int j = 0; j < q; ++j){
      arma::mat Xj = Xtilde.rows(j * n, (j+1) * n - 1);
      E2(i,j) = arma::trace(Xi * Beta_UQ * Xj.t());
    }
  }
  
  arma::mat S_post = arma::eye(q,q) + V.t()*V + E2;
  double df_post = q + n;
  
  Sigma = S_post / (df_post + q + 1); // mode of IW
  Q = arma::inv_sympd(Sigma);         
  Si = arma::chol(Q, "lower");       
  S = arma::inv(arma::trimatl(Si));
  
  // future
  //A = S.t();
  //Aplus = arma::pinv(A);
  //AplusT = Aplus.t();
}



double SpIOX::latent_fit_eval(){
  // for vi
  arma::mat E = Y - X * E_B - E_W;
  
  double ll = 0.0;
  for(int j=0; j<q; j++){
    arma::uvec ix = avail_by_outcome(j);
    
    double navail = .0 + avail_by_outcome(j).n_elem;
    
    arma::vec ej = E.col(j);
    arma::vec ej_sub = ej(ix);
    
    double ssq = arma::accu(arma::square(ej_sub));
    double dj = Ddiag(j); // diagonal element of D
    
    ll += -0.5 * navail * log(2.0 * M_PI * dj) - 0.5 * ssq / dj;
  }
  
  return ll;
}



double SpIOX::logdens_eval(){
  arma::vec vecYtilde = arma::vectorise(Y - X * B);
  arma::vec daggp_logdets = arma::zeros(q);
  for(unsigned int j=0; j<q; j++){
    arma::mat vYtlocal = vecYtilde.subvec(j*n, (j+1)*n-1);
    vecYtilde.subvec(j*n, (j+1)*n-1) = daggps.at(j).H_times_A(vYtlocal, daggp_use_H);
    daggp_logdets(j) = daggps.at(j).precision_logdeterminant;
  }
  
  arma::mat Ytildemat = arma::mat(vecYtilde.memptr(), n, q, false, true);
  arma::vec ytilde = arma::vectorise(Ytildemat * Si);
  
  double curr_ldet = +0.5*arma::accu(daggp_logdets);
  double curr_logdens = curr_ldet - 0.5*arma::accu(pow(ytilde, 2.0));
  
  return curr_logdens;
}

