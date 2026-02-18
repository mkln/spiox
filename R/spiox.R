#' Fit GP-IOX models for multivariate spatial outcomes
#'
#' @description
#' Fits the GP-IOX response or latent model using either MCMC or variational inference (VI).
#' A Vecchia DAG is constructed internally from `coords` and `m` (grid-aware if `coords` are gridded).
#'
#' @param Y Numeric matrix of responses with dimensions `n x q` (rows = locations, columns = outcomes).
#' @param X Numeric design matrix with dimensions `n x p` (must have the same number of rows as `Y`).
#' @param coords Numeric matrix of spatial coordinates with dimensions `n x 2` (must have the same number of rows as `Y`).
#' @param m Integer. Vecchia neighbor size used to construct the DAG.
#' @param method Character. One of `"response"` or `"latent"`.
#' @param fit Character. One of `"mcmc"` or `"vi"`.
#' @param iter Integer. Number of MCMC iterations if `fit="mcmc"`. If `fit="vi"`, this is the maximum number of iterations allowed.
#' @param print_every Integer. Print progress every `print_every` iterations. Set to `0` for silent.
#' @param starting Named list of starting values. Any omitted elements are initialized internally. Theta requires care:
#'   \itemize{
#'     \item `Beta`: numeric matrix `p x q` (starting regression coefficients, defaults to all `0`).
#'     \item `Sigma`: numeric matrix `q x q` (starting outcome covariance / coregionalization, defaults to `diag(q)`).
#'     \item `Theta`: numeric matrix `3 x q` with rows corresponding to `phi`, `nu`, `alpha`.
#'       `Theta` must be supplied when `fit="vi"` or when `opts$update_Theta = c(0,0,0)`.
#'     \item `Ddiag`: numeric vector of length `q` (latent-model only; defaults to `rep(1, q)`).
#'   }
#' @param opts Optional named list of algorithm/model options. Any omitted elements fall back to defaults.
#'   \itemize{
#'     \item `matern`: integer (0=squared exponential (nu is the exponent); 1=Matern (nu is smoothness)).
#'     \item `num_threads`: integer number of threads (OpenMP if available, defaults to `1`).
#'     \item `update_Theta`: integer vector of length 3 indicating which of (`phi`, `nu`, `alpha`) to update via MCMC.
#'       Defaults: latent MCMC `c(0,0,0)`; response MCMC `c(1,0,1)`. Ignored for `fit='vi'`.
#'     \item `tol`: numeric tolerance used for VI stopping.
#'   }
#' @param debug Optional named list of MCMC controls (primarily for development). Any omitted elements fall back to defaults.
#'   \itemize{
#'     \item `sampling`: integer specifying the latent-model sampler variant (only used when `method="latent"` and `fit="mcmc"`).
#'     \item `sample_Beta`: logical; sample regression coefficients in MCMC.
#'     \item `sample_Sigma`: logical; sample Sigma in MCMC.
#'     \item `sample_Ddiag`: logical; sample D in latent-model MCMC.
#'   }
spiox <- function(Y, X, coords, m=15, 
                  method = c("response","latent"),
                  fit = c("mcmc","vi"),
                  iter = 1000, print_every = 100, 
                  starting=NULL, opts=NULL, debug=NULL){
  
  method <- match.arg(method)
  fit    <- match.arg(fit)
  
  Y <- as.matrix(Y); X <- as.matrix(X); coords <- as.matrix(coords)
  
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X); d <- ncol(coords)
  stopifnot(nrow(X) == n, nrow(coords) == n, d == 2L)
  
  # ---- gridded + DAG ----
  is_gridded <- function(xy, tol = sqrt(.Machine$double.eps)) {
    x <- sort(unique(xy[,1])); y <- sort(unique(xy[,2]))
    dx <- diff(x); dy <- diff(y)
    ok_step <- function(d) length(d) <= 1L || max(d, na.rm=TRUE) <= tol || (max(d)-min(d) <= tol*max(1, max(d)))
    nx <- length(x); ny <- length(y)
    nrow(unique(xy)) == nx*ny && ok_step(dx) && ok_step(dy)
  }
  
  gridded <- is_gridded(coords)
  custom_dag <- dag_vecchia(coords, m, gridded)
  dag_opts <- if (gridded) -1L else 0L
  
  # ---- opts (method-specific defaults) ----
  default_update_theta <- if (method == "latent") c(0L,0L,0L) else c(1L,0L,1L)
  
  opts_defaults <- list(
    update_Theta = default_update_theta,
    num_threads  = 1L,
    tol          = 1e-2,
    matern       = 1
  )
  opts <- modifyList(opts_defaults, if (is.null(opts)) list() else opts)
  stopifnot(length(opts$update_Theta) == 3L)
  
  # must initialize theta if we're not updating it
  if (all(opts$update_Theta == 0L) && (is.null(starting) || is.null(starting$Theta))) {
    stop("starting$Theta must be supplied when update_Theta = c(0,0,0).")
  }
  # vi does not update theta: must supply
  if (fit == "vi" && (is.null(starting) || is.null(starting$Theta))) {
    stop("starting$Theta (3 x q with rows phi, nu, alpha) must be supplied for fit='vi'.")
  }
  
  # debug controls MCMC 
  debug_defaults <- list(
    sampling = 2L,  # 2 = n sequential, q block (current default)
    sample_Beta = TRUE,
    sample_Sigma = TRUE,
    sample_Ddiag = TRUE
  )

  debug <- modifyList(debug_defaults, if (is.null(debug)) list() else debug)
  
  # ---- starting values ----
  if (is.null(starting)) starting <- list()
  
  `%||%` <- function(a, b) if (!is.null(a)) a else b
  
  Beta_start  <- starting$Beta  %||% matrix(0, p, q)
  Sigma_start <- starting$Sigma %||% diag(q)
  
  # Theta: 3 rows (phi, nu, alpha), q columns
  Theta_user <- starting$Theta %||% NULL
  
  if (!is.null(Theta_user)) {
    Theta_user <- as.matrix(Theta_user)
    stopifnot(nrow(Theta_user) == 3L, ncol(Theta_user) == q)
  } else {
    # only allowed if you're updating at least one theta (and not VI),
    # due to checks above
    alpha_default <- ifelse(method == "latent", 0, 0.1)
    Theta_user <- rbind(10, 0.5, alpha_default)
  }
  rownames(Theta_user) <- c("phi","nu","alpha")
  
  # checking user supplied values
  if (method == "latent" && !is.null(Theta_user)) {
    alpha <- as.numeric(Theta_user[3, ])
    if (any(abs(alpha) > 0)) {
      warning("In method='latent', alpha (last row of starting$Theta) should be 0; nonzero values were supplied.")
    }
  }
  
  Theta_start <- rbind(
    phi   = Theta_user[1, , drop=FALSE],
    sigmasq = rep(1, q),                 # not used
    nu    = Theta_user[2, , drop=FALSE],
    alpha = Theta_user[3, , drop=FALSE]
  )
  update_Theta_full <- as.integer(c(opts$update_Theta[1], 0L, opts$update_Theta[2], opts$update_Theta[3]))
  
  
  # latent-only
  Ddiag_start <- starting$Ddiag %||% rep(1, q)
  
  # ---- dispatch ----
  out <- switch(paste(method, fit, sep=":"),
                "response:mcmc" = spiox_response(
                  Y, X, coords, custom_dag,
                  Beta_start, 
                  Sigma_start, 
                  Theta_start,
                  mcmc=iter, 
                  print_every=print_every, 
                  matern=opts$matern, 
                  dag_opts=dag_opts,
                  sample_Beta=debug$sample_Beta, 
                  sample_Sigma=debug$sample_Sigma,
                  update_Theta=update_Theta_full,
                  num_threads=as.integer(opts$num_threads)
                ),
                "latent:mcmc" = spiox_latent(
                  Y, X, coords, custom_dag,
                  Beta_start, 
                  Sigma_start, 
                  Theta_start, 
                  Ddiag_start,
                  mcmc=iter, 
                  print_every=print_every, 
                  matern=opts$matern, 
                  dag_opts=dag_opts,
                  sample_Beta=debug$sample_Beta, 
                  sample_Sigma=debug$sample_Sigma,
                  sample_Ddiag=debug$sample_Ddiag,
                  update_Theta=update_Theta_full,
                  num_threads=as.integer(opts$num_threads),
                  sampling=as.integer(debug$sampling)
                ),
                "response:vi" = spiox_response_vi(
                  Y, X, coords, custom_dag, dag_opts,
                  Theta=Theta_start, 
                  Sigma_start=Sigma_start, 
                  Beta_start=Beta_start,
                  print_every=print_every, 
                  matern=opts$matern,
                  num_threads=as.integer(opts$num_threads)
                ),
                "latent:vi" = 
                  spiox_latent_vi(
                  Y, X, coords, custom_dag, dag_opts,
                  Theta=Theta_start, 
                  Sigma_start=Sigma_start, 
                  Beta_start=Beta_start,
                  Ddiag_start=Ddiag_start,
                  matern=opts$matern, 
                  num_threads=as.integer(opts$num_threads),
                  print_every=print_every, 
                  tol=opts$tol, max_iter=iter
                ),
                stop("Unknown method/fit combo")
  )
  
  # drop internal sigma2 for consistency
  if("Theta" %in% names(out)){
    out$Theta <- out$Theta[-2,,,drop=F]
  }
  
  out$call    <- match.call()
  out$method  <- method
  out$fit     <- fit
  out$gridded <- gridded
  out$dag_opts <- dag_opts
  return(out)
}

