###############################################
# Run PC algorithm on all synthetic non-Gaussian NID datasets, with true moral graph
###############################################



# import libraries
library(igraph)
library(pcalg)

# import helper functions
source("/Users/tongxu/Downloads/projects/micodag/R_methods/helper_functions.R")


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
      
      p <- dim(X)[2]
      
      mat <- matrix(TRUE, p, p)
      for (i in 1:NROW(moral.graph)) {
        mat[moral.graph[i,1], moral.graph[i,2]] <- FALSE
        mat[moral.graph[i,2], moral.graph[i,1]] <- FALSE
      }
      
      # run
      start_time <- Sys.time()
      pc.fit = pc(suffStat = list(C=cor(X), n =400), indepTest = gaussCItest, alpha=0.01, labels = colnames(X), fixedGaps = mat)
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
      gdag0m = graph_from_adjacency_matrix(as(pc.fit, "amat"))
      edges = matrix(as.numeric(get.edgelist(gdag0m, names=TRUE)),ncol = 2)
      graph_pred = igraph.to.graphNEL(gdag0m)
      
      cpdag_ori <- dag2cpdag(graph_ori)
      cpdag_pred <- dag2cpdag(graph_pred)
      d_cpdag <- sum(abs(as(cpdag_ori, "matrix") - as(cpdag_pred, "matrix")))
      # SHD <- shd(graph_ori, graph_pred)
      # SHDs <- shds(graph_ori, graph_pred)
      # rates <- compare.Graphs(graph_ori, graph_pred)
      g1 = wgtMatrix(cpdag_ori, transpose = FALSE)
      g2 = wgtMatrix(cpdag_pred, transpose = FALSE)
      TPR = sum(g2[g1==1])/sum(g1)
      FPR = sum(g2[g1!=1])/(dim(g1)[1]**2 - sum(g2[g1==1]))
      result <- list(m=mm, alpha=alpha, k=kk, time=TIME, d_cpdag=d_cpdag, TPR=TPR, FPR=FPR)
      print(result)
      results = rbind(results, result)
    }
  }
}


# write the results into a csv file
write.csv(results, "./experiment results/comparison with synthetic non-Gaussian errors/1-30/pc_synthetic_non-Gaussian_results.csv",row.names=FALSE)
