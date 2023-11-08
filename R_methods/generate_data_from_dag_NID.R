# Generate non-equal variance data from dag
# This is for the experiment of early stopping where I need to generate DAGs and
# data in NID cases.


library(igraph)
library(MASS)


data.path = "E:\\Northwestern\\Research\\independent study 1\\dag\\gurobi\\Data\\SyntheticDataNID"
setwd(data.path)

eweights <- c(-0.8, -0.6, 0.6, 0.8)
sigvec <- c(0.5, 1, 1.5)
nsamples <- 400
ndata <- 10

m_list = c(10,20,30)
filenames <- list.files(data.path, "^D")  # DAG edge files starting with D.
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
  covmat <- diag(sample(sigvec, nv, replace=T))
  covmat <- infmat %*% covmat %*% t(infmat)
  
  ## generate data and write it into the same folder
  for(jj in 1:ndata){
    set.seed(jj)
    datmat <- mvrnorm(n=nsamples, mu=rep(0,nv), Sigma=covmat)
    
    datfilename <- paste0(
      paste("data","m",m, "n", nsamples, "iter", jj, sep="_"), ".csv")
    write.table(datmat, datfilename, sep = ",", 
                row.names=FALSE, col.names=FALSE)
  }
}