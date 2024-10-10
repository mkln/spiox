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

dag_vecchia_predict <- function(coords, newcoords, m=20){
  nn.found <- FNN::get.knnx(coords, newcoords, k=m)
  
  pred_dag <- as.list(as.data.frame(t(nn.found$nn.index-1)))
  return(pred_dag)
}