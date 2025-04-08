###############################################
# Run ges algorithm on all NID datasets with estimated moral graph
###############################################

# This script is for comparison between our algorithm and ges algorithm
# We will use pcalg

library(pcalg)
library(igraph)
library(glue)

source("/Users/tongxu/Downloads/projects/micodag/R_methods/helper_functions.R")

setwd("/Users/tongxu/Downloads/projects/micodag")
dataset.folder <- "/Users/tongxu/Downloads/projects/micodag/Data/RealWorldDatasetsTXu-NonGaussian/"
datasets <- list.files(dataset.folder)

#####################################
# Run for each dataset
#####################################

results <- data.frame()

for (dataset in datasets) {
  print(dataset)
  # collect file paths
  for (kk in c(1:10)) {
    data.file = list.files(paste(dataset.folder,dataset,sep="/"), paste0("n_500_iter_", kk))[1]
    true.graph.file = list.files(paste(dataset.folder,dataset,sep="/"), "Original")
    mgest.file = list.files(glue("{dataset.folder}{dataset}/"), glue("superstructure_glasso_iter_{kk}.txt"))
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Sparse_Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    est.moral.graph = read.table(paste(dataset.folder,dataset,mgest.file, sep="/"), header=FALSE, sep=",")
    est.moral.graph = as.matrix(est.moral.graph, dimnames = NULL)
    est.moral.graph = matrix(as.logical(est.moral.graph), nrow=dim(X)[2], ncol=dim(X)[2])
    p <- dim(X)[2]
    
    est.moral.graph = !est.moral.graph
    
    start_time <- Sys.time()
    # score = new("GaussL0penObsScore", data=X, lambda=2)
    score = new("GaussL0penObsScore", data=X)
    ges.fit <- ges(score, fixedGaps = est.moral.graph)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    # generate a graph object from original graph
    nodes = dim(X)[2]
    ori_gg <- make_empty_graph(n = nodes)  
    for (x in c(1:nrow(true.graph))){
      ori_gg <- ori_gg %>% add_edges(c(true.graph[x,1],true.graph[x,2]))
      
    }
    graph_ori = as_graphnel(ori_gg)
    
    # result analysis
    
    # generate a graph object from estimated graph
    graph_pred = as(ges.fit$essgraph,"graphNEL")
    
    cpdag_ori <- dag2cpdag(graph_ori)
    cpdag_pred <- dag2cpdag(graph_pred)
    d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
    # SHD <- shd(graph_ori, graph_pred)
    # SHDs <- shds(graph_ori, graph_pred)
    g1 = wgtMatrix(cpdag_ori, transpose = FALSE)
    g2 = wgtMatrix(cpdag_pred, transpose = FALSE)
    TPR = sum(g2[g1==1])/sum(g1)
    FPR = sum(g2[g1!=1])/(dim(g1)[1]**2 - sum(g2[g1==1]))
    
    result <- list(network=dataset,instance=kk, time=TIME, d_cpdag=d_cpdag, TPR=TPR, FPR=FPR)
    print(result)
    results = rbind(results, result)
  }
}
mean(results$d_cpdag)
write.csv(results, "ges_results_est_non-Gaussian.csv",row.names=FALSE)

