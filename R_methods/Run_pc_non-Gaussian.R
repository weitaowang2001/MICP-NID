###############################################
# Run pc algorithm on all NID datasets with non-Gaussian errors and true moral
###############################################

# This script is for comparison between our algorithm and pc algorithm
# We will use pcalg

library(pcalg)
library(igraph)

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
    # mgest.file = list.files(paste(dataset.folder,dataset,sep="/"), "mgest_PearsonCorEst.txt")
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Sparse_Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    
    p <- dim(X)[2]
    
    mat <- matrix(TRUE, p, p)
    for (i in 1:NROW(moral.graph)) {
      mat[moral.graph[i,1], moral.graph[i,2]] <- FALSE
      mat[moral.graph[i,2], moral.graph[i,1]] <- FALSE
    }
    
    
    start_time <- Sys.time()
    pc.fit = pc(suffStat = list(C=cor(X), n =500), indepTest = gaussCItest, alpha=0.01, labels = colnames(X), fixedGaps = mat)
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
    gdag0m = graph_from_adjacency_matrix(as(pc.fit, "amat"))
    edges = matrix(as.numeric(get.edgelist(gdag0m, names=TRUE)),ncol = 2)
    graph_pred = igraph.to.graphNEL(gdag0m)
    
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

write.csv(results, "pc_results_non-Gaussian.csv",row.names=FALSE)

