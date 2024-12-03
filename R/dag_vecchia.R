
#' Generate a Directed Acyclic Graph (DAG) for Vecchia Approximation
#'
#' This function creates a Directed Acyclic Graph (DAG) structure based on spatial coordinates 
#' for use in Vecchia approximations, which are commonly employed for scalable Gaussian process 
#' modeling. The DAG defines the conditional independence structure for the spatial locations.
#'
#' @param coords A numeric matrix or data frame (\eqn{n \times d}) of spatial coordinates, where \eqn{n} 
#' is the number of locations and \eqn{d} is the spatial dimension (e.g., 2 for latitude and longitude).
#' @param m An integer specifying the number of nearest neighbors to consider when constructing the DAG. 
#' Default is 20.
#'
#' @return A list where each element corresponds to a location in the input coordinates and contains 
#' the indices of its parent nodes in the DAG. These indices represent the nearest neighbors that 
#' define the conditional independence structure for each location. Indexing starts from 0 for input to other package functions.
#' The return object retains the original indexing of the input coordinates.
#' For returning with the maxmin ordering, use `dag_vecchia_maxmin`.
#' 
#' @examples
#' # Generate a DAG for a 2D spatial grid
#' coords <- matrix(runif(200), ncol = 2)  # 100 spatial locations in 2D
#' dag <- dag_vecchia(coords, m = 10)
#' 
#' @importFrom GPvecchia order_maxmin_exact
#' @importFrom GPvecchia findOrderedNN_kdtree2
#' @export
dag_vecchia <- function(coords, m=20){
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  ixmm_order <- order(ixmm)
  
  coords_mm <- coords[ixmm,]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  nn_dag_reorder <- (nn_dag %>% lapply(\(dag) ixmm[dag+1]-1 ))[ixmm_order] 
  return(nn_dag_reorder)
}


dag_vecchia_maxmin <- function(coords, m=20){
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  ixmm_order <- order(ixmm)
  
  coords_mm <- coords[ixmm,]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
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