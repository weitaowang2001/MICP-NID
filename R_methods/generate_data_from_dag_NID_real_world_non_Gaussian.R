##############
### 06/06/2024
### Generate 10 datasets for each real world network from 1dsep to 12hepar
### But use non-Gaussian errors.
### The idea follows from the JMLR paper by Shimizu 06. 
##############
library(igraph)
library(MASS)

data.path = "/Users/tongxu/Downloads/projects/micodag/Data/RealWorldDatasetsTXu-NonGaussian_30/"
setwd(data.path)


filenames <- list.files(data.path)

eweights <- c(-0.8, -0.6, 0.6, 0.8)
sigvec <- c(0.5, 1, 1.5)
nsamples <- 500
ndata <- 30


power_nonlinearity <- function(x) {
  # Define the intervals
  intervals <- list(c(0.5, 0.8), c(1.2, 2.0))
  
  # Randomly choose one of the intervals
  chosen_interval <- sample(intervals, 1)
  
  # Generate a random number within the chosen interval
  random_power <- runif(1, min = chosen_interval[[1]][1], max = chosen_interval[[1]][2])
  
  return(sign(x)*abs(x)**random_power)
}

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
  
  ## generate data and write it into the same folder
  for(jj in 11:ndata){
    set.seed(jj)
    errors <- mvrnorm(n=nsamples, mu=rep(0,nv), Sigma=covmat)
    errors1 <- sapply(errors, power_nonlinearity)
    errors1 <- matrix(errors1, nrow=nsamples, ncol=nv)
    datmat <- t(infmat %*% t(errors1))
    datfilename <- paste0(
      paste("data", fname, "n", nsamples, "iter", jj, sep="_"), ".csv")
    datfilename <- paste(data.path, fname, datfilename, sep="/")
    write.table(datmat, datfilename, sep = ",", 
                row.names=FALSE, col.names=FALSE)
  }
}