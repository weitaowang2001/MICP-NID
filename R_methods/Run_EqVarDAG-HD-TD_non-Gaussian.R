###############################################
# Run EqVarDAG-HD-TD on all NID datasets with non-Gaussian errors
###############################################

# This script is for comparison between our algorithm and EqVarDAG-HD-TD
# We will use EqVarDAG_HD_TD.R


# import libraries
library(igraph)
library(pcalg)

# import helper functions
source("/Users/tongxu/Downloads/projects/micodag/R_methods/helper_functions.R")
source("/Users/tongxu/Downloads/projects/micodag/R_methods/EqVarDAG_HD_TD.R")


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
    
    # run
    start_time <- Sys.time()
    result <- EqVarDAG_HD_TD(X)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    # generate a graph object from original graph
    nodes = dim(X)[2]
    ori_gg <- make_empty_graph(n = nodes)  
    for (x in c(1:nrow(true.graph))){
      ori_gg <- ori_gg %>% add_edges(c(true.graph[x,1],true.graph[x,2]))
      
    }
    graph_ori = igraph.to.graphNEL(ori_gg)
    
    # result analysis
    
    # generate a graph object from estimated graph
    gdag0m = graph_from_adjacency_matrix(result$adj, mode = "directed", weighted = NULL, diag = TRUE, add.colnames = NULL, add.rownames = NA)
    edges = matrix(as.numeric(get.edgelist(gdag0m, names=TRUE)),ncol = 2)
    graph_pred = igraph.to.graphNEL(gdag0m)
    
    cpdag_ori <- dag2cpdag(graph_ori)
    cpdag_pred <- dag2cpdag(graph_pred)
    d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
    SHD <- shd(graph_ori, graph_pred)
    SHDs <- shds(graph_ori, graph_pred)
    rates <- compare.Graphs(graph_ori, graph_pred)
    
    result <- list(network=dataset,instance=kk, time=TIME, d_cpdag=d_cpdag, SHD=SHD, SHDs=SHDs, TPR=rates$TPR, FPR=rates$FPR)
    print(result)
    results = rbind(results, result)
  }
}

# # write the results into a text file
# mylist_str <- sapply(names(results), 
#                  function(name) paste(name, paste(names(results[[name]]), 
#                 unlist(results[[name]]), sep = "=", collapse = ","), sep = ":"))
# writeLines(mylist_str, EqVarDAG-HD-TD_results.txt")

# write the results into a csv file
write.csv(results, "EqVarDAG-HD-TD_results_non-Gaussian.csv",row.names=FALSE)
