##############
### 12/07/2024
### Generate 10 datasets for each real world network from 1dsep to 12hepar
### Most part of this script is copied from "datfromgraph_unequalvar.R"
### equal variances
##############
library(igraph)
library(MASS)

data.path = "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasets_ID/"
setwd(data.path)


filenames <- list.files(data.path)

eweights <- c(-0.8, -0.6, 0.6, 0.8)
sigvec <- c(1, 1, 1)
nsamples <-500
ndata <- 10

#ii=3
for(ii in 1:length(filenames)){
  fname = filenames[ii]
  
  ## read the edge list 
  filename <- list.files(paste(data.path, fname, sep="/"),"Sparse_Original")
  elist <- read.csv(paste(data.path, fname, filename, sep="/"))
  
  ## create a graph object and get adjacency matrix
  gg <- graph_from_edgelist(as.matrix(elist))
  adjmat <- t(as.matrix(get.adjacency(gg)))
  nv <- ncol(adjmat)
  
  ## add weights to the adjacency matrix and obtain influence matrix
  set.seed(ii)
  adjmat_wgtd <- adjmat * 
    matrix(sample(eweights, nv*nv, replace=T), nv, nv) 
  Ip <- diag(1, nv, nv)
  infmat <- solve(Ip - adjmat_wgtd)
  
  ## covariance matrix for random noise with non-equal variance
  ## using formulas in Shojaie & Michailidis (2010)
  set.seed(ii)
  covmat <- diag(sample(sigvec, nv, replace=T))
  covmat <- infmat %*% covmat %*% t(infmat)
  
  ## generate data and write it into the same folder
  for(jj in 1:ndata){
    set.seed(jj)
    datmat <- mvrnorm(n=nsamples, mu=rep(0,nv), Sigma=covmat)
    
    datfilename <- paste0(
      paste("data", fname, "n", nsamples, "iter", jj, sep="_"), ".csv")
    datfilename <- paste(data.path, fname, datfilename, sep="/")
    write.table(datmat, datfilename, sep = ",", 
                row.names=FALSE, col.names=FALSE)
  }
}