# test performance of GES for reordered data

library(pcalg)
library(igraph)
library(glue)

source("/Users/tongxu/Downloads/projects/MICODAG-CD/R_methods/helper_functions.R")

setwd("/Users/tongxu/Downloads/projects/MICODAG-CD")
dataset.folder <- "/Users/tongxu/Downloads/projects/MICODAG-CD/Data/RealWorldDatasetsTXu_smallalpha/"


datasets <- list.files(dataset.folder)

results <- data.frame()

for (dataset in datasets) {
  for (kk in c(1:10)) {
    data.file = list.files(paste(dataset.folder,dataset,sep="/"), glue("n_500_iter_{kk}"))[1]
    true.graph.file = list.files(paste(dataset.folder,dataset,sep="/"), "Original")
    mgest.file = list.files(glue("{dataset.folder}/{dataset}/"), glue("superstructure_glasso_iter_{kk}.txt"))
    true.moral.file = list.files(paste(dataset.folder,dataset,sep="/"), "Sparse_Moral")
    
    X = as.matrix(read.csv(paste(dataset.folder,dataset, data.file, sep="/"), header=FALSE))
    true.graph = read.table(paste(dataset.folder,dataset,true.graph.file, sep="/"), header=FALSE, sep=",")
    moral.graph = read.table(paste(dataset.folder,dataset,true.moral.file, sep="/"), header=FALSE, sep=",")
    estimated.moral = as.matrix(read.table(paste(dataset.folder,dataset,mgest.file, sep="/"), header=FALSE, sep=","))
    estimated.moral = !estimated.moral
    
    random_ordering = sample(1:ncol(X))
    X = X[,random_ordering]
    estimated.moral <- estimated.moral[random_ordering,random_ordering]
    
    true.graph_adj <- matrix(0, nrow = ncol(X), ncol = ncol(X))
    for (x in c(1:nrow(true.graph))){
      true.graph_adj[true.graph[x,1],true.graph[x,2]] = 1
    }
    
    true.graph_adj <- true.graph_adj[random_ordering,random_ordering]
    
    ori_gg = graph_from_adjacency_matrix(true.graph_adj)
    graph_ori = as_graphnel(ori_gg)
    
    
    start_time <- Sys.time()
    # score = new("GaussL0penObsScore", data=X, lambda=2)
    score = new("GaussL0penObsScore", data=X)
    ges.fit <- ges(score, fixedGaps = estimated.moral)
    end_time <- Sys.time()
    TIME <- as.numeric(end_time - start_time, units="secs")
    
    Omega <- ges.fit$repr$err.var()
    B <- ges.fit$repr$weight.mat()
    Gamma <- (diag(ncol(X)) - B)%*%diag(Omega**(-1/2))
    obj = sum(-2*(log(diag(Gamma)))) + sum(diag(Gamma%*%t(Gamma)%*%cov(X))) + log(nrow(X))/2/nrow(X)*(sum(Gamma != 0) - ncol(X))
    
    # # generate a graph object from original graph
    # nodes = dim(X)[2]
    # ori_gg <- make_empty_graph(n = nodes)
    # for (x in c(1:nrow(true.graph))){
    #   ori_gg <- ori_gg %>% add_edges(c(true.graph[x,1],true.graph[x,2]))
    # 
    # }
    # graph_ori = as_graphnel(ori_gg)
    
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
    
    result <- list(dataset=dataset,k=kk, Time=TIME, d_cpdag=d_cpdag, TPR=TPR, FPR=FPR)
    print(result)
    results = rbind(results, result)
  }
  
}

write.csv(results, "./experiment results/comparison with benchmarks/ges_results_est_default_reorder.csv",row.names=FALSE)


