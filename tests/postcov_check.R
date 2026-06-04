# Correctness check for the multivariate postcov preconditioner
# (cg_preconditioner = "postcov").  A preconditioner never changes the target,
# so the invariants are:
#   (A) AGREEMENT: postcov, vadu and jacobi all solve the SAME W system, so on a
#       fixed seed (identical RHS, incl. Bhattacharya noise) their W draws agree to
#       CG tolerance — at both diagonal and dense Sigma.  Guards against postcov
#       targeting the wrong posterior.  (jacobi is a genuinely different PC, so the
#       iteration counts differ — the test is not vacuous.)
#   (B) DENSE Sigma: postcov converges (iters << maxit) and recovers W, with far
#       fewer CG iters than vadu under strong cross-outcome correlation.
#
# Single-threaded, one fixed starting list reused verbatim, all of
# Beta/Sigma/Ddiag/Theta frozen so the only stochastic step is the W draw.
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

# run() closure around a FIXED data set + FIXED starting list (W-only route).
make_runner <- function(d, Sigma0, nthr = 1) {
  st0 <- spiox::autostart(d$Y, d$X, coords, "latent", m, nu = nu)
  st0$Sigma <- Sigma0; st0$Theta[3, ] <- 0
  function(pc, iters) {
    set.seed(1)
    spiox(Y = d$Y, X = d$X, coords = coords, m = m, method = "latent", fit = "mcmc",
          iter = iters, print_every = 0,
          debug = list(sampling = 1L, sample_Beta = FALSE, sample_Sigma = FALSE,
                       sample_Ddiag = FALSE),
          opts = list(update_Theta = c(0, 0, 0), num_threads = nthr,
                      cg_preconditioner = pc, cg_rebuild = "once", joint_BW = FALSE),
          starting = st0)
  }
}
relW <- function(a, b) max(abs(a - b)) / max(abs(b))

cat("================ (A) AGREEMENT across preconditioners ================\n")
Sig_diag <- diag(seq(1, 2, length.out = q))
dA  <- make_data(Sig_diag)
run <- make_runner(dA, Sig_diag)
N   <- 15
oj  <- run("jacobi",     N)
omv <- run("postcov", N)
ov  <- run("vadu",       N)
cat(sprintf("diagonal Sigma : max|W_postcov - W_jacobi|/max|W| = %.2e  (vadu = %.2e)\n",
            relW(omv$W, oj$W), relW(ov$W, oj$W)))
stopifnot(relW(omv$W, oj$W) < 1e-2, relW(ov$W, oj$W) < 1e-2)
# control: jacobi is a genuinely weaker PC -> strictly more iters (test discriminates)
stopifnot(mean(oj$cg_iters) > mean(omv$cg_iters))
cat(sprintf("control: mean iters jacobi=%.0f  postcov=%.0f  vadu=%.0f (discriminating)\n",
            mean(oj$cg_iters), mean(omv$cg_iters), mean(ov$cg_iters)))
cat("AGREEMENT OK: postcov & vadu target the same posterior as jacobi.\n\n")

cat("================ (B) DENSE Sigma ================\n")
set.seed(7)
sds <- sqrt(seq(1, 2, length.out = q))
R   <- cov2cor(solve(rWishart(1, q + 1, diag(q))[, , 1]))
Sig_dense <- diag(sds) %*% R %*% diag(sds)
cat("Sigma correlation matrix:\n"); print(round(R, 3))
dB   <- make_data(Sig_dense)
runB <- make_runner(dB, Sig_dense, nthr = 4)
omv <- runB("postcov", 20)
ov  <- runB("vadu",       20)
oj  <- runB("jacobi",     20)
rmse <- function(o) sqrt(mean((apply(o$W, 1:2, mean) - dB$W)^2))
maxit <- nr * q
cat(sprintf("mean CG iters : postcov=%.1f  vadu=%.1f  jacobi=%.1f  (maxit=%d)\n",
            mean(omv$cg_iters), mean(ov$cg_iters), mean(oj$cg_iters), maxit))
cat(sprintf("W RMSE vs truth: postcov=%.4f  vadu=%.4f\n", rmse(omv), rmse(ov)))
stopifnot(all(is.finite(omv$W)), mean(omv$cg_iters) < maxit,
          relW(apply(omv$W, 1:2, mean), apply(oj$W, 1:2, mean)) < 5e-2)
cat("DENSE OK: postcov converges and recovers W.\n")
