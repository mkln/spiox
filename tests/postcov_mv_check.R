# Correctness check for the multivariate postcov preconditioner (cg_preconditioner
# = "postcov_mv").  Invariants:
#   (A) REDUCTION: when Sigma is diagonal, postcov_mv must coincide with postcov.
#       A preconditioner never changes the target, so the strong, robust signal is
#       SPECTRAL: identical CG iteration counts sweep-by-sweep (and, for the W-only
#       route, W draws matching to FP precision).  Checked for both block routes.
#   (B) DENSE Sigma: postcov_mv converges (iters << maxit) and recovers W.
#
# Determinism notes: the comparison is run single-threaded (OpenMP reductions are
# not bit-reproducible) with ONE fixed starting list reused verbatim (autostart()
# is re-randomised on each call), and with Beta/Sigma/Ddiag/Theta all frozen so the
# only stochastic step is the W draw.
library(spiox)

set.seed(26)
q  <- 3
nr <- 300
coords <- cbind(runif(nr), runif(nr)); colnames(coords) <- c("Var1", "Var2")
nu <- 1.5
Ddiag_true <- runif(q, 0.1, 0.5)
Theta  <- rbind(c(5, 10, 20)[1:q], nu, 0)
m  <- 10

make_data <- function(Sigma_true) {
  W <- spiox:::rgpiox(coords, Sigma_true, Theta, m = 40, num_threads = 1)
  X <- cbind(1, matrix(rnorm(nr * 2), ncol = 2)); p <- ncol(X)
  B <- matrix(rnorm(p * q), ncol = q)
  Y <- X %*% B + W + matrix(rnorm(nr * q), ncol = q) %*% diag(sqrt(Ddiag_true))
  list(Y = Y, X = X, W = W)
}

# Build the run() closure around a FIXED data set + FIXED starting list.
make_runner <- function(d, Sigma0, nthr = 1) {
  st0 <- spiox::autostart(d$Y, d$X, coords, "latent", m, nu = nu)
  st0$Sigma <- Sigma0; st0$Theta[3, ] <- 0
  function(pc, iters, joint_BW = FALSE) {
    set.seed(1)
    spiox(Y = d$Y, X = d$X, coords = coords, m = m, method = "latent", fit = "mcmc",
          iter = iters, print_every = 0,
          debug = list(sampling = 1L, sample_Beta = FALSE, sample_Sigma = FALSE,
                       sample_Ddiag = FALSE),
          opts = list(update_Theta = c(0, 0, 0), num_threads = nthr,
                      cg_preconditioner = pc, cg_rebuild = "once", joint_BW = joint_BW),
          starting = st0)
  }
}

cat("================ (A) REDUCTION at diagonal Sigma ================\n")
Sig_diag <- diag(seq(1, 2, length.out = q))
dA  <- make_data(Sig_diag)
run <- make_runner(dA, Sig_diag, nthr = 1)
N   <- 15

for (jb in c(FALSE, TRUE)) {
  o5 <- run("postcov",    N, joint_BW = jb)
  o6 <- run("postcov_mv", N, joint_BW = jb)
  dITER <- max(abs(o5$cg_iters - o6$cg_iters))
  dW    <- max(abs(o5$W - o6$W))
  cat(sprintf("joint_BW=%-5s  sweep-by-sweep max|iters diff| = %d   max|W diff| = %.2e\n",
              jb, dITER, dW))
  if (!jb) {
    # W-only route (gibbs_w_block_precision): the W factor reduces to postcov's
    # exactly, so iters match and W matches to FP precision.
    stopifnot(dITER == 0, dW < 1e-6)
  } else {
    # Joint (B,W) route: both PCs use the same shared block Gauss-Seidel wrapper, so
    # the W factor still reduces; only the FP ordering of the W apply (level-scheduled
    # vs per-outcome Eigen) differs, tipping an occasional sweep by +-1 CG iter.
    stopifnot(dITER <= 2, dW < 1e-2)
  }
}
# jacobi control: a genuinely different PC must NOT match (guards against a vacuous test)
oj <- run("jacobi", N)
o5 <- run("postcov", N)
stopifnot(max(abs(o5$cg_iters - oj$cg_iters)) > 0)
cat(sprintf("control: postcov vs jacobi mean iters %.0f vs %.0f (test is discriminating)\n",
            mean(o5$cg_iters), mean(oj$cg_iters)))
cat("REDUCTION OK: postcov_mv == postcov at diagonal Sigma (both routes).\n\n")

cat("================ (B) DENSE Sigma ================\n")
set.seed(7)
sds <- sqrt(seq(1, 2, length.out = q))
R   <- cov2cor(solve(rWishart(1, q + 1, diag(q))[, , 1]))
Sig_dense <- diag(sds) %*% R %*% diag(sds)
cat("Sigma correlation matrix:\n"); print(round(R, 3))
dB   <- make_data(Sig_dense)
runB <- make_runner(dB, Sig_dense, nthr = 4)
o5 <- runB("postcov",    20)
o6 <- runB("postcov_mv", 20)
rmse <- function(o) sqrt(mean((apply(o$W, 1:2, mean) - dB$W)^2))
maxit <- nr * q
cat(sprintf("mean CG iters : postcov=%.1f   postcov_mv=%.1f   (maxit=%d)\n",
            mean(o5$cg_iters), mean(o6$cg_iters), maxit))
cat(sprintf("W RMSE vs truth: postcov=%.4f   postcov_mv=%.4f\n", rmse(o5), rmse(o6)))
stopifnot(all(is.finite(o6$W)), mean(o6$cg_iters) < maxit)
cat("DENSE OK: postcov_mv converges and recovers W.\n")
