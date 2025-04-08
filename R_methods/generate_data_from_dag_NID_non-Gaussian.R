# Generate non-equal variance data from dag. Use non-Gaussian errors.
# This is for the experiment of testing non-Gaussian errors on small graphs.


library(igraph)
library(MASS)


data.path = "/Users/tongxu/Downloads/projects/micodag/Data/SyntheticDataNID/"
setwd(data.path)

alphas <- c(4)
eweights <- c(-0.8, -0.6, 0.6, 0.8)
# sigvec <- c(0.5, 1, 1.5)
nsamples <- 400
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


m_list = c(10,15,20)
# filenames <- list.files(data.path, "^D")  # DAG edge files starting with D.
for (alpha in alphas) {
  for (i in c(1:length(m_list))){
    m = m_list[i]
    fname = list.files(data.path, paste0("^DAG_",m))
    elist = read.table(fname) # read edge list
    
    ## create a graph object and get adjacency matrix
    gg <- graph_from_edgelist(as.matrix(elist))
    adjmat <- t(as.matrix(get.adjacency(gg)))
    nv <- ncol(adjmat)
    
    ## add weights to the adjacency matrix and obtain influence matrix
    set.seed(i)
    adjmat_wgtd <- adjmat * 
      matrix(sample(eweights, nv*nv, replace=T), nv, nv) 
    Ip <- diag(1, nv, nv)
    infmat <- solve(Ip - adjmat_wgtd)
    
    ## covariance matrix for random noise with non-equal variance
    ## using formulas in Shojaie & Michailidis (2010)
    set.seed(i)
    # covmat <- diag(sample(sigvec, nv, replace=T))
    covmat <- diag(runif(nv, 4-alpha, 4+alpha))
    # covmat <- infmat %*% covmat %*% t(infmat)
    
    ## generate data and write it into the same folder
    for(jj in 11:ndata){
      set.seed(jj)
      errors <- mvrnorm(n=nsamples, mu=rep(0,nv), Sigma=covmat)
      errors1 <- sapply(errors, power_nonlinearity)
      errors1 <- matrix(errors1, nrow=nsamples, ncol=nv)
      datmat <- t(infmat %*% t(errors1))
      datfilename <- paste0(
        paste("./non-Gaussian/data","m",m, "n", nsamples, "alpha", alpha, "iter", jj, sep="_"), ".csv")
      write.table(datmat, datfilename, sep = ",", 
                  row.names=FALSE, col.names=FALSE)
    }
  }
  
}