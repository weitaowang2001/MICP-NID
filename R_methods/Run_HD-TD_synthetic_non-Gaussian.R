###############################################
# Run EqVarDAG-HD-TD on all synthetic non-Gaussian NID datasets
###############################################



# import libraries
library(igraph)
library(pcalg)

# import helper functions
source("/Users/tongxu/Downloads/projects/micodag/R_methods/helper_functions.R")
source("/Users/tongxu/Downloads/projects/micodag/R_methods/EqVarDAG_HD_TD.R")


setwd("/Users/tongxu/Downloads/projects/micodag")
dataset.folder <- "/Users/tongxu/Downloads/projects/micodag/Data/SyntheticDataNID_30/"

#####################################
# Run for each dataset
#####################################

results <- data.frame()


for (mm in c(10, 15, 20)) {
  print(mm)
  for (alpha in c(4)) {
    # collect file paths
    for (kk in c(1:30)) {
      data.file = list.files(paste0(dataset.folder,"/non-Gaussian"), paste0("data_m_", mm, "_n_400_alpha_", alpha, "_iter_", kk))[1]
      true.graph.file = paste0("DAG_", mm, ".txt")
      true.moral.file = paste0("Moral_DAG_", mm, ".txt")
      
      X = as.matrix(read.csv(paste0(dataset.folder,"non-Gaussian/",data.file), header=FALSE))
      true.graph = read.table(paste0(dataset.folder, true.graph.file), header=FALSE, sep=" ")
      moral.graph = read.table(paste0(dataset.folder, true.moral.file), header=FALSE, sep=" ")
      
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
      
      result <- list(m=mm,alpha=alpha,k=kk, time=TIME, d_cpdag=d_cpdag, SHD=SHD, SHDs=SHDs, TPR=rates$TPR, FPR=rates$FPR)
      print(result)
      results = rbind(results, result)
    }
  }
}


# write the results into a csv file
write.csv(results, "./experiment results/comparison with synthetic non-Gaussian errors/1-30/EqVarDAG-HD-TD_synthetic_non-Gaussian_results.csv",row.names=FALSE)
