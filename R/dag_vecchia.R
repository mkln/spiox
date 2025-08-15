#' Construct a DAG for Vecchia Approximation
#'
#' Builds a directed acyclic graph (DAG) representing the conditional dependence
#' structure used in Vecchia approximations for Gaussian process models.
#' 
#' For non-gridded data, locations are ordered via the max–min ordering, and
#' parents are selected as the \code{m} nearest previously ordered locations.
#' For gridded data (\code{gridded=TRUE}), coordinates are assumed to lie on a
#' Cartesian grid; ordering and parent sets are chosen to maximize reuse of
#' identical parent–parent covariance blocks across locations.
#'
#' @param coords Numeric matrix (\eqn{n \times d}) of coordinates, with \eqn{n} locations
#' and \eqn{d} spatial dimensions.
#' @param m Integer, number of nearest neighbors (parents) per node. Default is 20.
#' @param gridded Logical. If \code{TRUE}, use grid-based ordering and grouping to
#' enable efficient exemplar caching; otherwise, use max–min ordering.
#'
#' @return A list of length \eqn{n}, where element \code{i} contains the 0-based indices
#' of the parent nodes for location \code{i}. These indices follow the original
#' ordering of \code{coords}.
#'
#' @examples
#' # Random points
#' coords <- matrix(runif(200), ncol=2)
#' dag <- dag_vecchia(coords, m=10)
#'
#' # Gridded points
#' gx <- expand.grid(x=1:10, y=1:10)
#' dag_g <- dag_vecchia(as.matrix(gx), m=5, gridded=TRUE)
#'
#' @export
dag_vecchia <- function(coords, m=20, gridded=FALSE){
  if(gridded){
    dag_for_gridded_cols(coords, m)
  } else {
    ixmm <- MaxMincpp(coords)
    ixmm_order <- order(ixmm)
    
    coords_mm <- coords[ixmm,]
    nn_dag_mat <- findOrderedNN_kdtree2(coords_mm, m)
    nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
    nn_dag_reorder <- (nn_dag %>% lapply(\(dag) ixmm[dag+1]-1 ))[ixmm_order] 
    return(nn_dag_reorder)
  }
}


dag_vecchia_maxmin <- function(coords, m=20){
  ixmm <- MaxMincpp(coords)
  ixmm_order <- order(ixmm)
  
  coords_mm <- coords[ixmm,]
  nn_dag_mat <- findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  nn_dag_reorder <- (nn_dag %>% lapply(\(dag) ixmm[dag+1]-1 ))[ixmm_order] 
  return(list(dag=nn_dag, maxmin=ixmm))
}


#' Generate a Directed Acyclic Graph (DAG) for predictions
#'
#' @param coords A numeric matrix or data frame (\eqn{n \times d}) of training spatial coordinates, where \eqn{n} 
#' is the number of locations and \eqn{d} is the spatial dimension (e.g., 2 for latitude and longitude).
#' @param newcoords A numeric matrix or data frame (\eqn{n_{o} \times d}) of testing spatial coordinates, where \eqn{n_o} 
#' is the number of locations and \eqn{d} is the spatial dimension (e.g., 2 for latitude and longitude).
#' @param m An integer specifying the number of nearest neighbors to consider when constructing the DAG. 
#' Default is 20.
#'
#' @return The DAG for predictions

#' @importFrom FNN get.knnx
#' @export
dag_vecchia_predict <- function(coords, newcoords, m=20){
  nn.found <- FNN::get.knnx(coords, newcoords, k=m)
  
  pred_dag <- as.list(as.data.frame(t(nn.found$nn.index-1)))
  return(pred_dag)
}