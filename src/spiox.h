#include "omp_import.h"
#include "sparse_solvers.h"
#include "daggp.h"
#include "ramadapt.h"

#include <memory>
#include <functional>

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
  unsigned int n, q, p, nq, pq, npq;
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
  //   PROBE     : adaptive default.  Two-candidate, q-keyed: the PRIMARY anchor
  //               (POSTCOV_MV if q>10 else POSTCOV) runs an uncapped probe_burnin
  //               and its worst-case iter count sets the cap; the SECONDARY (the
  //               other) is then trialed probe_trial sweeps capped at that, and
  //               adopted only if it converges within the cap with a lower mean
  //               than the primary; otherwise the primary is kept.  See the
  //               probe-state block below.
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
    PRECOND_VADU      = 4,
    PRECOND_POSTCOV   = 5,
    PRECOND_POSTCOV_MV = 6
  };
  PrecondChoice precond_choice = PRECOND_PROBE;

  // How often the Σ/Ddiag-dependent preconditioner factors are rebuilt during
  // MCMC.  A preconditioner only accelerates CG and never shifts the sampled
  // target, so freezing it at the autostart values ("once") is exact, while
  // rebuilding every sweep ("always") keeps it tracking the drifting operator at
  // extra cost.
  //   REBUILD_AUTO   : per-PC default — ALWAYS for POSTCOV (its rebuild is O(nq)
  //                    cheap), ONCE for POSTERIOR (expensive FSAI factor of A_j)
  //                    and RESPONSE (expensive C+D Vecchia refactor).
  //   REBUILD_ALWAYS : rebuild every sweep regardless of PC.
  //   REBUILD_ONCE   : build once at the autostart values, then freeze.
  // VADU is a special case: its factor always rebuilds (see pc_rebuild_now)
  // regardless of this setting, because the recompute is trivially cheap.
  enum RebuildMode { REBUILD_AUTO = 0, REBUILD_ALWAYS = 1, REBUILD_ONCE = 2 };
  int cg_rebuild = REBUILD_AUTO;
  // When true, the POSTERIOR and POSTCOV preconditioners drop the cross-outcome
  // Σ-mix (R_corr) and apply only their per-outcome (block-diagonal) factors:
  //   block(j,k) = δ_{jk}·(per-outcome A_j^{-1} approx).
  // This isolates the local-Σ/Ddiag tracking of cg_rebuild="always" from the
  // cross-outcome coupling (which can hurt CG conditioning when Σ is strongly
  // correlated).  No effect on VADU/RESPONSE/JACOBI, nor on the sampling=3
  // per-outcome sequential sampler (which is already uncoupled).
  bool pc_diagonal = false;
  // Resolve cg_rebuild for PC `pc` given how many times its factors have already
  // been built (`builds_done`): returns whether to (re)build this sweep.
  bool pc_rebuild_now(PrecondChoice pc, int builds_done) const {
    // VADU always rebuilds: its factor is an O(nq) trivial recompute (dscale /
    // R_corr), so freezing it saves nothing and a stale dscale only hurts CG.
    // This overrides cg_rebuild = "once" (now the global default) for VADU only.
    if(pc == PRECOND_VADU) return true;
    int mode = cg_rebuild;
    // POSTCOV and POSTCOV_MV default to ALWAYS: their rebuild is cheap (POSTCOV is
    // an O(nq) coefficient rescale; POSTCOV_MV overwrites preallocated q×q block
    // values in place, parallel over locations) and tracking the live Σ/Ddiag
    // empirically beats freezing at the autostart values.
    if(mode == REBUILD_AUTO)
      mode = (pc == PRECOND_POSTCOV || pc == PRECOND_POSTCOV_MV) ? REBUILD_ALWAYS : REBUILD_ONCE;
    return (mode == REBUILD_ALWAYS) ? true : (builds_done == 0);
  }

  // Probe state.  Two-candidate q-keyed probe (used by both the sampling=1 block
  // sampler and the sampling=3 per-outcome sampler via probe_step):
  //   Phase 0 (burn-in): run the PRIMARY anchor — POSTCOV_MV if q>10 else POSTCOV —
  //     for probe_burnin sweeps (uncapped) and record its worst-case (max) iters,
  //     which becomes the iteration cap for the secondary.
  //   Phase 1 (trial)   : run the SECONDARY (the other of POSTCOV/POSTCOV_MV) for
  //     probe_trial sweeps, capped at the primary max; disqualified if it hits the
  //     cap without converging.
  //   Decision          : adopt the secondary only if it converged within the cap
  //     with a lower mean than the primary; otherwise keep the primary.
  // All probe sweeps are real MCMC draws (only the preconditioner differs).
  int           probe_count         = 0;           // sweeps done (burn-in + trial)
  int           probe_cap           = 0;           // secondary iter cap = primary max
  double        probe_primary_sum   = 0.0;         // primary burn-in cumulative iters
  int           probe_primary_max   = 0;           // primary burn-in worst-case iters
  double        probe_secondary_sum = 0.0;         // secondary trial cumulative iters
  bool          probe_secondary_ok  = true;        // secondary converged within the cap?
  static constexpr int probe_burnin = 10;          // primary burn-in sweeps (probe length)
  static constexpr int probe_trial  = 5;           // secondary trial sweeps
  // One probe step: `run(pc, cap)` performs a real W-sampling sweep with PC `pc`
  // (cap = maxit, 0 = uncapped) and returns its CG iteration count.  Advances the
  // probe state machine, sets precond_choice when the winner is locked in, fills
  // cg_iter, and returns the PC actually used this sweep.
  PrecondChoice probe_step(int& cg_iter,
                           const std::function<int(PrecondChoice, int)>& run);

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

  // Per-outcome state for the POSTCOV PC.  "Latent posterior-conditional"
  // preconditioner: instead of a Vecchia factor of the posterior PRECISION
  // (VAPOP), we build a cheap Vecchia factor of the posterior COVARIANCE using
  // single-datum approximate conditionals
  //     p(w_i | w_{N_i}, y) ≈ p(w_i | w_{N_i}, y_i)
  //                         ∝ p(w_i | w_{N_i}) · p(y_i | w_i).
  // Per outcome j, reusing the PRIOR daggps[j] coefficients b_i = daggps[j].h(i)
  // and r_i = daggps[j].sqrtR(i)^2, with invDj = 1/Ddiag(j) (0 at missing):
  //     g_i = Q_jj / (Q_jj + r_i·invDj)         (shrinkage toward data, ∈(0,1])
  //     f_i = r_i  / (Q_jj + r_i·invDj)          (conditional variance)
  // The lower-tri factor Ũ_j has rows [1/√f_i at i, -g_i·b_i/√f_i at parents]
  // so that Ũ_jᵀŨ_j ≈ A_j (the posterior precision), applied as A_j⁻¹ ≈
  // Ũ_j⁻¹Ũ_j⁻ᵀ by two triangular solves, then mixed across outcomes by R_corr.
  // Build is O(nq) cheap (just rescales prior coefs), so it rebuilds every sweep.
  int postcov_pc_n_builds = 0;
  std::vector<Eigen::SparseMatrix<double>> postcov_U_eigen;   // size q (col-major lower)
  std::vector<Eigen::SparseMatrix<double>> postcov_Ut_eigen;  // size q (col-major upper, = transpose)
  // Build the latent posterior-conditional factors Ũ_j (rebuilt every sweep).
  void build_postcov_factors();

  // Per-location state for the MULTIVARIATE POSTCOV PC ("postcov_mv").  Instead
  // of q separate per-outcome factors mixed by a uniform R_corr, this builds a
  // SINGLE block-Vecchia inverse-Cholesky factor U of the JOINT posterior
  // covariance, with q×q blocks coupling all outcomes at a location.  The IOX
  // prior has V = HW with rows v_i = (V_{i1..iq}) iid N(0,Σ), so the multivariate
  // prior conditional is
  //     w_i | w_{N_i} ~ N(M_i w_{N_i}, R_i),  R_i = Λ_i Σ Λ_i,
  // with Λ_i = diag_j sqrtR_i^{(j)} (the per-outcome conditional sd, from
  // daggps[j].sqrtR(i)) and M_i's parent-t block = diag_j b_i^{(j)}(t) (from
  // daggps[j].h(i)(t); the shared parents are daggps[0].dag(i)).  Folding the
  // local datum y_i (precision Ω_i = diag_j invD_ij, 0 at missing) gives the
  // posterior conditional covariance F_i = (R_i^{-1}+Ω_i)^{-1} = L_i L_iᵀ and the
  // mean coefficient G_i = F_i R_i^{-1} M_i.  The block factor U (location-major,
  // lower-block-triangular) has diagonal block L_i^{-1} and parent-t block
  // -L_i^{-1} G_i^{(t)}, so UᵀU ≈ joint posterior precision and M^{-1} = U^{-1}U^{-ᵀ}.
  // Stored as dense q×q blocks (NOT an Eigen sparse matrix — for dense q×q blocks
  // the BLAS gemv beats a scalar sparse triangular solve, measured 2–3× at q=10–30):
  //     postcov_mv_L[i] = L_i                          (q×q lower)
  //     postcov_mv_B[i] = [ -L_i^{-1}G_i^{(1)} | ... ] (q × q·m_i; empty if root)
  // Reduces EXACTLY to POSTCOV when Σ is diagonal.  Per-rebuild work O(n(q³+m q²))
  // (rows independent → OpenMP-parallel), reusing R_i^{-1}=Λ_i^{-1}QΛ_i^{-1} (Q=Σ^{-1}
  // is a member, so only one q×q SPD inverse per location).
  //
  // Apply M^{-1}=U^{-1}U^{-ᵀ} is a block triangular solve — inherently serial along
  // the DAG.  We parallelise it by LEVEL SCHEDULING: level[i] = 1+max level over
  // parents; locations within a level are mutually independent, so each level is an
  // OpenMP-parallel sweep (forward = levels up, gathering from parents; back = levels
  // down, gathering from children).  Both passes run in place on one q×n buffer.
  // The level/children structure is DAG-only (θ/Σ/Ddiag-independent) → precomputed
  // once.  This is the lever postcov lacked: location- (n-) parallelism rather than
  // the per-outcome q-parallelism, where postcov_mv's far fewer CG iters pays off.
  int  postcov_mv_n_builds = 0;
  bool postcov_mv_setup_done = false;             // levels/children/storage precomputed?
  std::vector<arma::mat> postcov_mv_L;            // size n, each q×q lower-triangular
  std::vector<arma::mat> postcov_mv_B;            // size n, each q×(q·m_i) (empty if root)
  arma::uvec             postcov_mv_order;        // locations sorted by level (stable)
  arma::uvec             postcov_mv_level_ptr;    // size n_levels+1: level slices into order
  arma::field<arma::uvec> postcov_mv_child_k;     // per i: children k (i is a parent of k)
  arma::field<arma::uvec> postcov_mv_child_t;     // per i: position of i in k's parent list
  arma::mat              postcov_mv_buf;          // reused q×n location-major apply buffer
  std::vector<arma::vec> postcov_mv_acc;          // per-thread scratch, length q
  std::vector<arma::vec> postcov_mv_zpar;         // per-thread scratch, length q·max_parents
  void postcov_mv_setup();                        // one-time levels + children + storage
  void build_postcov_mv_factors();
  // Apply the postcov_mv W-half PC to one W-block: z = U^{-1}U^{-ᵀ} r, r_w / z_w
  // length nq (outcome-major).  Level-scheduled parallel block substitution.
  void postcov_mv_apply(const double* r_w, double* z_w);

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
  void gibbs_w_sequential_byoutcome(int& cg_iter, PrecondChoice precond,
                                    int cg_maxit_override = 0);

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
    
    nq = n*q;
    pq = p*q;
    npq = pq + nq;
    
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
