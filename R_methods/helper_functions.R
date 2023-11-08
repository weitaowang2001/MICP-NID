######################################
# This file contains helper functions for analyzing the results from
# EqVarDAG-HD-BU, EqVarDAG-HD-TD, and EqVarDAG-TD.



shds <- function(g1, g2)
  #' Calculate the SHDs of the true graph and estimated graph
  #' This function is copied from the source code of shd(), and removing the
  #' reversal counting part
{
  if (is(g1, "pcAlgo")) 
    g1 <- g1@graph
  if (is(g2, "pcAlgo")) 
    g2 <- g2@graph
  if (is(g1, "graphNEL")) {
    m1 <- wgtMatrix(g1, transpose = FALSE)
    m1[m1 != 0] <- 1
  }
  if (is(g2, "graphNEL")) {
    m2 <- wgtMatrix(g2, transpose = FALSE)
    m2[m2 != 0] <- 1
  }
  shd <- 0
  s1 <- m1 + t(m1)
  s2 <- m2 + t(m2)
  s1[s1 == 2] <- 1
  s2[s2 == 2] <- 1
  ds <- s1 - s2
  ind <- which(ds > 0)
  m1[ind] <- 0
  shd <- shd + length(ind)/2
  ind <- which(ds < 0)
  m1[ind] <- m2[ind]
  shd <- shd + length(ind)/2
  return(shd)
}

compare.Graphs <- function(g1, g2)
  #' calculate the TPR and FPR
  #' g1 is the true graph, and g2 is estimated graph
{
  if (is(g1, "graphNEL")) {
    m1 <- wgtMatrix(g1, transpose = FALSE)
    m1[m1 != 0] <- 1
  }
  if (is(g2, "graphNEL")) {
    m2 <- wgtMatrix(g2, transpose = FALSE)
    m2[m2 != 0] <- 1
  }
  s1 = m1 + t(m1)
  s1[s1 == 2] <- 1
  P = sum(m1)  # number of positive
  TP = sum(s1[m2 == 1] == 1)  # number of true positive
  TPR = TP/P
  
  FP = sum(s1[m2 == 1] != 1)
  N = dim(m1)[1]**2 - P
  
  FPR = FP/N
  return(list(TPR=TPR, FPR=FPR))
}
