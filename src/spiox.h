#include "omp_import.h"
#include "sparse_solvers.h"
#include "daggp.h"
#include "ramadapt.h"

#include <memory>

using namespace std;

int time_count(std::chrono::steady_clock::time_point tstart);

class SpIOX {
public:
  // Y = XB + Z where Z is multivariate q,q IOC-GP
  // S^T*S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1. Lambda, Q are sparse
  
  // -------------- data
  
  // matrix of outcomes dim n, q 
  arma::mat Y;
  // matrix of predictors dim n, p
  arma::mat X;
  
  // metadata
  unsigned int n, q, p;
  int intercept; // location of intercept in X, if at all
  int num_threads;
  //double spatial_sparsity;
  
  // -------------- model parameters
  // objects that depend on B
  arma::mat B;
  arma::mat YXB;
  arma::mat B_Var; // prior variance on B, element by element
  
  // objects that depend on Sigma
  arma::mat S, Si, Sigma, Q; // S^T * S = Sigma = Q^-1 = (Lambda*Lambda^T + Delta)^-1 = (Si * Si^T)^-1
  
  // future:
  //arma::mat A, Aplus, AplusT; 
  
  // objects that depend on theta RadGP for spatial dependence
  std::vector<DagGP> daggps, daggps_alt;
  arma::mat theta; // each column is one alternative value for theta
  bool gridded;
  
  // objects that depend on W
  arma::mat V; 
  
  // -------------- utilities
  int matern;
  void update_B(); // 
  void update_Sigma_iwishart();
  
  void compute_V(); 
  bool upd_theta_metrop();
  arma::uvec upd_theta_metrop_conditional(); // returns uvec with changes to thetaj
  void init_theta_adapt();
  
  // data with with misalignment
  arma::field<arma::uvec> avail_by_outcome;
  arma::umat missing_mat;
  arma::uvec rows_with_missing;
  arma::uvec rows_some;
  int n_some; // n where at least one is nonmissing
  bool Y_needs_filling;
  arma::uvec Y_na_indices;
  void manage_missing_data(){
    // managing misalignment, i.e. not all outcomes observed at all locations
    // indices of non-NAs 
    avail_by_outcome = arma::field<arma::uvec>(q);
    missing_mat = arma::zeros<arma::umat>(n, q);
    arma::uvec row_miss_01 = arma::zeros<arma::uvec>(n);
    arma::uvec row_complete_miss_01 = arma::zeros<arma::uvec>(n);
    Y_na_indices = arma::find_nonfinite(Y);
    for(int j=0; j<q; j++){
      avail_by_outcome(j) = arma::find_finite(Y.col(j));
      for(int i=0; i<n; i++){
        missing_mat(i,j) = !arma::is_finite(Y(i,j));
      }
    }
    Y.elem(arma::find(missing_mat)).zeros(); // set missing to zero
    
    for(int i=0; i<n; i++){
      row_miss_01(i) = arma::any(missing_mat.row(i) == 1);
      row_complete_miss_01(i) = arma::all(missing_mat.row(i) == 1); 
    }
    rows_with_missing = arma::find(row_miss_01 == 1);
    rows_some = arma::find(row_complete_miss_01 == 0);
    n_some = rows_some.n_elem;
    if(n_some < n){
      if(latent_model != 1){
        Rcpp::stop("Fully missing rows in Y detected. Use latent_model=1 or exclude from data and rerun.\n");
      }
    }
    //if(Y.has_nonfinite() & (latent_model==1)){
    //  Rcpp::stop("nq block not implemented for misaligned data.\n");
    //}
    if(arma::accu(missing_mat)>0){
      Y_needs_filling = true;
    } else {
      Y_needs_filling = false;
    }
  }
  void sample_Y_misaligned(const arma::uvec& theta_changed);
  
  // Preconditioner / W-sampler choices for the block latent model.
  // PROBE (the default) auto-selects among POSTERIOR, RESPONSE, and VADU.
  //
  //   JACOBI    : diagonal of the joint operator (precision form).
  //               O(nq + pq) per apply; nearly free; tends to be a weak PC.
  //
  //   POSTERIOR : block-diagonal-on-(B,W) PC with cross-outcome Σ-mix on
  //               the W half.  B half: per-outcome exact dense Cholesky of
  //                 M_B,j = diag(1/B_Var.col(j)) + X^T·diag(invD.col(j))·X
  //               W half: blkdiag(H_A,j^T)·(R_corr ⊗ I)·blkdiag(H_A,j)
  //                 with R_corr = D_σ^{-1}·Σ·D_σ^{-1}, the correlation matrix
  //                 of Σ.  H_A,j is the per-outcome Vecchia precision factor
  //                 of A_j = Q_jj·H_j^T H_j + diag(invD.col(j)), built once
  //                 per chain via local regression on (m+1)x(m+1) sub-blocks
  //                 of A_j and mirrored as col-major sparse Eigen for the
  //                 fast double-mult apply.
  //               R_corr chosen so the PC equals A_W^{-1} exactly in the
  //               D→0 limit (modulo Vecchia); reduces to a per-outcome block
  //               PC at Σ diagonal.  Extra cost vs the diagonal-Σ version:
  //               one n×q · q×q dense multiply per CG iter (cheap for q small).
  //
  //   PROBE     : adaptive default.  Compares {POSTERIOR, RESPONSE, VADU}
  //               (Jacobi excluded) over probe_per_pc sweeps each, then locks
  //               in the fastest.  A reference candidate runs first, uncapped,
  //               and sets a CG-iter budget; the rest run capped at the
  //               reference's worst case and are disqualified if they hit the
  //               cap without converging.  Reference = RESPONSE when Y is fully
  //               observed, POSTERIOR when Y has missing entries (the response
  //               C+D solve degrades under misalignment).  Winner = converged
  //               candidate with the fewest mean iters (ties to the reference).
  //               See the probe-state block below.
  //   RESPONSE  : not a PCG branch.  Selects an alternative w-block sampler
  //               (gibbs_w_block_marginal) that samples W in the covariance
  //               (data) domain via the Bhattacharya algorithm, solving
  //               systems with the marginal C+D rather than the precision
  //               C^{-1}+D^{-1}.  Preconditioned by a fresh per-outcome
  //               Vecchia factor of C+D (built with nugget = Ddiag).
  //
  //   VADU      : "Vecchia approximation with diagonal update" (Kündig &
  //               Sigrist).  PCG branch on the same precision system as
  //               JACOBI/POSTERIOR.  Reuses the prior Vecchia factor H_j and
  //               folds the likelihood diagonal into it:
  //                 P_VADU,j = B_j^T (W_j + Q_jj D_j^{-1}) B_j,
  //               with H_j = D_j^{-1/2} B_j (B_j unit lower-tri, D_j = R_j).
  //               Apply P^{-1} = two prior triangular solves + a diagonal
  //               scale by 1/(R_j⊙w_j + Q_jj), plus the same R_corr Σ-mix
  //               as POSTERIOR.  No per-sweep factor build (unlike VAPOP).
  enum PrecondChoice {
    PRECOND_PROBE     = 0,
    PRECOND_JACOBI    = 1,
    PRECOND_POSTERIOR = 2,
    PRECOND_RESPONSE  = 3,
    PRECOND_VADU      = 4
  };
  PrecondChoice precond_choice = PRECOND_PROBE;

  // How often the Σ/Ddiag-dependent preconditioner factors are rebuilt during
  // MCMC.  A preconditioner only accelerates CG and never shifts the sampled
  // target, so freezing it at the autostart values ("once") is exact, while
  // rebuilding every sweep ("always") keeps it tracking the drifting operator at
  // extra cost.
  //   REBUILD_AUTO   : per-PC default — ALWAYS for VADU (its rebuild is O(nq)
  //                    cheap: dscale / R_corr / per-outcome B Cholesky), ONCE for
  //                    POSTERIOR (expensive FSAI factor of A_j) and RESPONSE
  //                    (expensive C+D Vecchia refactor).
  //   REBUILD_ALWAYS : rebuild every sweep regardless of PC.
  //   REBUILD_ONCE   : build once at the autostart values, then freeze.
  enum RebuildMode { REBUILD_AUTO = 0, REBUILD_ALWAYS = 1, REBUILD_ONCE = 2 };
  int cg_rebuild = REBUILD_AUTO;
  // Resolve cg_rebuild for PC `pc` given how many times its factors have already
  // been built (`builds_done`): returns whether to (re)build this sweep.
  bool pc_rebuild_now(PrecondChoice pc, int builds_done) const {
    int mode = cg_rebuild;
    if(mode == REBUILD_AUTO) mode = (pc == PRECOND_VADU) ? REBUILD_ALWAYS : REBUILD_ONCE;
    return (mode == REBUILD_ALWAYS) ? true : (builds_done == 0);
  }

  // Probe state.  The probe compares at most three candidate preconditioners —
  // {POSTERIOR, RESPONSE, VADU} — over probe_per_pc sweeps each, then locks in
  // the winner.  Jacobi is deliberately excluded (it stays a user-selectable
  // option but never participates in auto-selection).
  //
  // Candidate ordering depends on the data:
  //   - any missing data  : reference = POSTERIOR (the marginal C+D response
  //                          solve degrades when the selection operator kicks
  //                          in), then RESPONSE, then VADU.
  //   - fully observed     : reference = RESPONSE (covariance-domain Bhattacharya
  //                          is typically fastest with aligned data), then
  //                          POSTERIOR, then VADU.
  // The reference candidate (probe_order[0]) runs uncapped and sets the iter
  // budget; every other candidate runs with maxit capped at the reference's
  // worst-case (max) iteration count.  A candidate that hits the cap without
  // converging is disqualified.  The winner is the converged candidate with the
  // lowest mean iteration count (ties resolved in favour of the reference).
  std::vector<PrecondChoice> probe_order;          // candidates, reference first
  int           probe_count   = 0;                 // sweeps done so far (across all candidates)
  int           probe_cap     = 0;                 // iter cap applied to non-reference candidates
  std::vector<double> probe_iter_sum;              // per-candidate cumulative iters
  std::vector<int>    probe_iter_max;              // per-candidate worst-case iters
  std::vector<bool>   probe_converged;             // per-candidate: still converging within budget?
  static constexpr int probe_per_pc = 5;

  // Per-outcome state for the POSTERIOR PC.  VAPOP = "Vecchia Approximation
  // of the POsterior Precision": for each outcome j we approximate
  //   A_j^{-1} ≈ H_A,j^T · H_A,j
  // where A_j = Q_jj·H_j^T H_j + diag(invD.col(j)) is the conditional
  // posterior precision of W_j | W_{-j}, Y_j.  H_A,j is built once via local
  // Vecchia regression on (m+1)x(m+1) sub-blocks of A_j and mirrored as
  // col-major sparse Eigen for the fast double-mult apply.  Idempotent
  // build guarded by vapop_n_builds.
  int vapop_n_builds = 0;
  std::vector<arma::vec>              vapop_sqrtR;     // size q, each length n
  std::vector<arma::field<arma::vec>> vapop_h;         // size q, each length n
  arma::field<arma::umat>             vapop_children;  // length n, each rows = [k, t_of_i_in_parents_of_k]
  std::vector<Eigen::SparseMatrix<double>> vapop_H_eigen;   // size q (col-major lower)
  std::vector<Eigen::SparseMatrix<double>> vapop_Ht_eigen;  // size q (col-major upper, = transpose)
  // How the POSTERIOR W-half preconditioner factor of A_j = Q_jj·HᵀH + D⁻¹ is built:
  //   0 = matrix-free  : assemble A_j entries on demand via DAG children-merge
  //                      walk (A_at), local FSAI solve -> bounded-m factor.
  //   1 = precision    : form P = HᵀH explicitly once (Eigen sparse product),
  //                      read its entries for the SAME local FSAI solve.
  // Both methods produce identical bounded-m factors (vapop_H_eigen).  The
  // local FSAI solve is the factored-sparse-approximate-inverse / Vecchia
  // inverse-Cholesky factor on the DAG sparsity pattern (see build_vapop_factors).
  int vapop_build_method = 0;
  // Build the Vecchia/FSAI factors of A_j (no-op if already built).  Called
  // once on the first POSTERIOR apply.
  void build_vapop_factors();

  // Telemetry: number of CG iterations used in the most-recent W-block update,
  // plus an integer code for which preconditioner / sampler ran
  // (0 unset / 1 jacobi / 2 posterior / 3 response / 4 vadu).  Read by the
  // outer MCMC driver and surfaced back to R.
  int last_cg_iter      = 0;
  int last_precond_used = 0;
  // Wall-clock seconds spent on the one-time ("once" mode) preconditioner
  // factor build (vapop / marginal-Vecchia / BW PC factors).  Accumulated the
  // first time the PC is constructed; 0 thereafter.  Surfaced back to R.
  double pc_build_seconds = 0.0;
  
  // latent model
  int latent_model; // 0: response, 1: block, 2: row seq, 3: col seq
  // For the block latent model (latent_model = 1):
  //   joint_BW = true  (default): sample (B, W) jointly via gibbs_BW_block.
  //   joint_BW = false          : blocked route — B|W (conjugate update_B),
  //                               then W|B in the precision domain
  //                               (gibbs_w_block_precision) or covariance domain
  //                               (RESPONSE), followed by an ASIS non-centred B
  //                               refresh (update_BW_asis), mirroring the
  //                               latent_model 2/3 samplers.  Honours the same
  //                               POSTERIOR / VADU / RESPONSE / JACOBI choices.
  bool joint_BW = true;
  arma::mat W;
  void w_sequential_singlesite(const arma::uvec& theta_changed);

  // Per-outcome sequential sampler for W (latent_model = 3).  Uses a fixed
  // sparse-precision-form CG with Jacobi PC (pcg_diag_solve) — the precond
  // enum is accepted for ABI compatibility but ignored.
  void gibbs_w_sequential_byoutcome(int& cg_iter, PrecondChoice precond);

  // β-conditional W update (ASIS reparameterisation) used by the non-block
  // latent samplers (latent_model = 2, 3) when sample_Beta is on.  Carries
  // its own inline Jacobi PC — doesn't touch the PrecondChoice enum.
  void update_BW_asis(int& cg_iter, arma::mat& B, arma::mat& W, bool sampling);

  // Joint BW PCG sampler — the only block sampler that survives the cleanup.
  // PC dispatched on `precond`: PRECOND_JACOBI (diagonal of the joint
  // precision) or PRECOND_POSTERIOR (block-diagonal-on-(B,W) with cross-
  // outcome Σ-mix on the W half via VAPOP factors).  `sampling` toggles
  // between MCMC sampling mode (RHS includes Bhattacharya noise terms) and
  // posterior-mean mode (deterministic).  `cg_maxit_override` > 0 caps the
  // PCG iter count (used by the probe phase to budget JACOBI against
  // POSTERIOR's measured iter count); 0 means use the default cap (n).
  void gibbs_BW_block(int& cg_iter, PrecondChoice precond, bool sampling=true,
                      int cg_maxit_override=0);

  // W | B precision-domain block sampler (the W-only counterpart of
  // gibbs_BW_block, used by the blocked route when joint_BW = false).  Holds B
  // fixed (sampled separately by update_B + ASIS), solving the conditional
  // W precision P_WW = Λ_W + diag(invD) via matrix-free PCG.  PC dispatched on
  // `precond`: PRECOND_JACOBI / PRECOND_POSTERIOR (VAPOP) / PRECOND_VADU — the
  // same W-half preconditioners as gibbs_BW_block, with no B half.  `sampling`
  // toggles Bhattacharya noise on the RHS; `cg_maxit_override` > 0 caps the PCG
  // iters (used by the probe), 0 = default cap (n).
  void gibbs_w_block_precision(int& cg_iter, PrecondChoice precond,
                               bool sampling=true, int cg_maxit_override=0);

  // Response covariance-form W-block sampler (PRECOND_RESPONSE).  Samples W as
  // a block via the Bhattacharya algorithm in the data domain: solves systems
  // with the marginal C+D (not the precision C^{-1}+D^{-1}) by matrix-free
  // PCG, preconditioned by a fresh per-outcome Vecchia factor of C+D (built
  // with nugget = Ddiag) sandwiched with the same R_corr Σ-mix as POSTERIOR.
  // B is sampled separately (conjugate) before this call.  Missing data
  // (misalignment) is handled via the selection-Bhattacharya reformulation:
  // a diagonal projection P (mask of observed entries) restricts the solve to
  // the observed block (M = ΦCΦᵀ + D_o), with the latent field at missing
  // entries kriged from the same draw — efficient when missingness is light.
  std::vector<DagGP> daggps_marginal;     // per-outcome Vecchia factor of C_j+D_j
  // "Once" mode: the marginal Vecchia factor is built a single time from the
  // starting theta + autostart Ddiag and then frozen.  A preconditioner only
  // affects CG convergence speed, never the sampled target, so freezing it at
  // the autostart values is exact and removes the per-sweep rebuild cost.
  int marginal_n_builds = 0;
  void build_marginal_daggps();           // build daggps_marginal once (nugget=Ddiag)
  // cg_maxit_override > 0 caps the PCG iteration count (used by the probe to
  // budget the RESPONSE candidate against the reference's measured iters);
  // 0 means use the default cap (n).
  void gibbs_w_block_marginal(int& cg_iter, bool sampling=true,
                              int cg_maxit_override=0);

  // Frozen POSTERIOR/VADU preconditioner buffers (built once, "once" mode).
  // chol_MBj : per-outcome p×p Cholesky of the B-half precision.
  // bw_R_corr: q×q correlation matrix of Σ used as the W-half mid-mix.
  // vadu_dscale: per-outcome sqrt(R_j⊙invD_j + Q_jj) diagonal scale (VADU only).
  int bw_pc_n_builds = 0;
  // VADU-specific build counter (kept separate from bw_pc_n_builds, which guards
  // POSTERIOR's FSAI build): tracks how many times the VADU factors have been
  // (re)built so cg_rebuild = "once" can freeze them.
  int vadu_pc_n_builds = 0;
  std::vector<arma::mat> bw_chol_MBj;
  arma::mat              bw_R_corr;
  std::vector<arma::vec> bw_vadu_dscale;

  arma::vec Ddiag;
  
  // centering of W and move to intercept (if there is one)
  void W_centering();
  // update running means
  void update_running_means(arma::mat&, const arma::mat&, bool pr=false); 
  
  // utilities for gibbs
  void update_BWSigma_px();
  void update_Sigma_gibbs();
  void update_Ddiag_gibbs();
  
  // whitened Y and X (that is, applying the same operation that makes W white noise)
  arma::mat Ytilde;
  arma::cube HX;
  
  // utilities for vi
  int vi_min_iter; // burn-in for vi in latent models
  int vi_it; // internal iteration counter
  //void update_B_vi();
  void update_Sigma_vi();
  void update_Ddiag_vi();
  bool vi;
  //arma::mat B_post_cov;
  arma::mat VTV, VTV_ma; // ma for moving average
  arma::mat ETE, ETE_ma;  
  arma::mat E_B; 
  arma::mat W_RB;
  arma::mat E_W;
  
  void vi_Beta_UQ(); // for computing Beta_UQ
  arma::vec delta_t, beta_running_mean;
  arma::mat Beta_UQ;
  arma::vec Ddiag_UQ;
  arma::mat Sigma_UQ;
  
  
  // utility for latent model and misaligned response model
  arma::field<arma::mat> Rw_no_Q;
  arma::field<arma::mat> Pblanket_no_Q;
  void cache_blanket_comps(const arma::uvec& theta_changed);
  
  // which theta updates are we doing
  bool phi_sampling, sigmasq_sampling, nu_sampling, alpha_sampling;
  
  // adaptive metropolis to update theta atoms
  int theta_mcmc_counter;
  arma::uvec which_theta_elem;
  arma::mat theta_unif_bounds;
  //arma::mat theta_metrop_sd;
  RAMAdapt theta_adapt;
  bool theta_adapt_active;
  // adaptive metropolis (conditional update) to update theta atoms
  // assume shared covariance functions and unknown parameters across variables
  arma::mat c_theta_unif_bounds;
  std::vector<RAMAdapt> c_theta_adapt;
  // --------
  
  
  // -------------- run 1 gibbs iteration based on current values
  void response_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq=false);
  void latent_gibbs(int it, int sample_sigma, bool sample_beta, bool update_theta, bool sample_tausq=false);
  void response_vi();
  void latent_vi();
  double latent_fit_eval();
  
  std::chrono::steady_clock::time_point tstart;
  arma::vec timings;
  
  // -------------- constructor for building C (dense! beware)
  SpIOX(const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int dag_opts,
        const arma::mat& daggp_theta, 
        const arma::mat& Sigma,
        int cov_model_matern,
        int num_threads_in){
    
    theta = daggp_theta;
    
    q = theta.n_cols;
    n = _coords.n_rows;
    
    daggps = std::vector<DagGP>(q);

    matern = cov_model_matern;

    // make n_threads depend on whether data are gridded, since behavior is opposite
    gridded = dag_opts==-1;
    int daggp_n_threads = gridded ? 1 : num_threads;
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag,
                        dag_opts,
                        matern,
                        daggp_n_threads);
    }
  }
  
  // -------------- constructor
  SpIOX(const arma::mat& _Y,
        const arma::mat& _X,
        const arma::mat& _coords,
        const arma::field<arma::uvec>& custom_dag,
        int dag_opts,
        int latent_model_choice,
        const arma::mat& Beta_start,
        const arma::mat& W_start,
        const arma::mat& Sigma_start,
        const arma::mat& daggp_theta,
        const arma::uvec& update_theta_which,
        const arma::vec& tausq_start,
        int cov_model_matern,
        int num_threads_in,
        int _vi_min_iter = 100)
  {
    num_threads = num_threads_in;

    Y = _Y;
    X = _X;

    n = Y.n_rows;
    q = Y.n_cols;
    p = X.n_cols;
    
    // intercept? 
    intercept = -1;
    for(int j=0; j<q; j++){
      if(arma::all(X.col(j) == 1.0)){
        intercept = j;
        break;
      }
    }
    
    latent_model = latent_model_choice; // latent model?

    manage_missing_data();
    
    if(latent_model>0){
      W = W_start;
      W(arma::find_nonfinite(W)).fill(0);
      E_W = W;
      W_RB = W;
      Ddiag = tausq_start;
    }
    
    
    // mcvi params
    vi = _vi_min_iter > 0;
    vi_min_iter = _vi_min_iter;
    vi_it = 0;
    
    VTV = arma::zeros(q, q);
    ETE = arma::zeros(q, q);
    delta_t = arma::zeros(p*q);
    beta_running_mean = arma::zeros(p*q);
    Beta_UQ = arma::zeros(p*q, p*q);
    
    VTV_ma = arma::zeros(q, q);
    ETE_ma = arma::zeros(q, q);
    
    Ytilde = arma::zeros(n, q);
    HX = arma::zeros(n, p, q);
    
    // continue with other init
    B = Beta_start;
    E_B = B;
    YXB = Y - X * B;
    
    B_Var = 1000 * arma::ones(arma::size(B));
    
    theta = daggp_theta;
    daggps = std::vector<DagGP>(q);

    // if multiple nu options, interpret as wanting to sample smoothness for matern
    // otherwise, power exponential with fixed exponent.
    phi_sampling = update_theta_which(0) == 1;
    sigmasq_sampling = update_theta_which(1) == 1;
    nu_sampling = update_theta_which(2) == 1;
    alpha_sampling = update_theta_which(3) == 1;

    matern = cov_model_matern;

    // make n_threads depend on whether data are gridded, since behavior is opposite
    gridded = dag_opts==-1;
    int daggp_n_threads = gridded ? 1 : num_threads;
    for(unsigned int i=0; i<q; i++){
      daggps[i] = DagGP(_coords, theta.col(i), custom_dag,
                               dag_opts,
                               matern,
                               daggp_n_threads);
    }
    daggps_alt = daggps;

    init_theta_adapt();
    
    S = arma::chol(Sigma_start, "upper");
    Si = arma::inv(arma::trimatu(S));
    Sigma = S.t() * S;
    Q = Si * Si.t();
    
    // future
    //A = S.t();
    //Aplus = arma::pinv(A);
    //AplusT = Aplus.t();
    
    compute_V();
    
    int nfill = latent_model == 2 ? n : rows_with_missing.n_elem;
    Rw_no_Q = arma::field<arma::mat> (nfill);
    Pblanket_no_Q = arma::field<arma::mat> (nfill);
    
    arma::uvec updater = arma::ones<arma::uvec>(q);
    if(latent_model | Y_needs_filling){
      // first time making markov blanket cache
      cache_blanket_comps(updater);
    }
    
    timings = arma::zeros(10);
  }
};
