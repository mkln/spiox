library(tidyverse)
library(Matrix)
library(scico)
set.seed(1)
nulist <- c(.5, .7, 1.5, 1)
q <- length(nulist)

Sigma <- solve(rWishart(1, q+1, diag(q))[,,1])

x <- seq(0,1, length.out=4)
coords <- expand.grid(x,x)
cx <- as.matrix(coords)
nr <- nrow(coords)
In <- diag(nr)

C <- 1:q %>% lapply(\(j) 
                    spiox::Correlationc(cx,cx,theta = c(1,1,j,0), covar = 1, T))


# iox
L <- C %>% lapply(\(Cm) t(chol(Cm)))
Lbig <- Matrix::bdiag(L)
C_iox <- as(Lbig %*% (Sigma %x% In) %*% t(Lbig), "CsparseMatrix")
image(C_iox)

# lmc 
A <- t(chol(Sigma))
C_lmc <- as((A %x% In) %*% Matrix::bdiag(C) %*% (t(A) %x% In), "CsparseMatrix")


ggplot_matrix_image <- function(M, label=NULL, full_symm=F){
  M <- M %>% as("CsparseMatrix")
  M_df <- as.data.frame(summary(M))
  
  colnames(M_df) <- c("row", "col", "value") # Renaming the columns for clarity
  
  if(full_symm){
    M_df <- bind_rows(M_df, M_df %>% 
                        rename(rowx=row, colx=col) %>% 
                        rename(row=colx, col=rowx)) %>% unique()
  }
  # Create the ggplot
  plotmade <- ggplot(M_df, aes(x = col, y = row, fill = value)) +
    geom_raster() +
    scale_fill_scico(palette="batlowK") +
    scale_y_reverse() + # Reverse the y-axis to match matrix convention
    theme_void() +
    labs(title = label,
         fill = "Value") +
    theme(legend.position="none",
          panel.border = element_rect(fill=NA, color="black"))
  return(plotmade)
}

iox_overall <- ggplot_matrix_image(C_iox)
iox_left <- ggplot_matrix_image(Lbig)
iox_center <- ggplot_matrix_image(Sigma %x% In, NULL, T)
iox_right <- ggplot_matrix_image(t(Lbig))

iox_visual <- gridExtra::grid.arrange(iox_overall, iox_left, iox_center, iox_right, nrow=1)


lmc_overall <- ggplot_matrix_image(C_lmc)
lmc_left <- ggplot_matrix_image(A %x% In)
lmc_center <- ggplot_matrix_image(Matrix::bdiag(C), NULL, T)
lmc_right <- ggplot_matrix_image(t(A) %x% In)

lmc_visual <- gridExtra::grid.arrange(lmc_overall, lmc_left, lmc_center, lmc_right, nrow=1)


iox_vs_lmc <- gridExtra::grid.arrange(iox_visual, lmc_visual, ncol=1)

ggsave(filename="figures/iox_overall.pdf", plot=iox_overall, width=3, height=3)
ggsave(filename="figures/iox_left.pdf", plot=iox_left, width=3, height=3)
ggsave(filename="figures/iox_center.pdf", plot=iox_center, width=3, height=3)
ggsave(filename="figures/iox_right.pdf", plot=iox_right, width=3, height=3)

ggsave(filename="figures/lmc_overall.pdf", plot=lmc_overall, width=3, height=3)
ggsave(filename="figures/lmc_left.pdf", plot=lmc_left, width=3, height=3)
ggsave(filename="figures/lmc_center.pdf", plot=lmc_center, width=3, height=3)
ggsave(filename="figures/lmc_right.pdf", plot=lmc_right, width=3, height=3)

