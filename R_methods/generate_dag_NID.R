# Using this 
library(pcalg)
library(igraph)
library(gRbase)

data.path = "E:\\Northwestern\\Research\\independent study 1\\dag\\gurobi\\Data\\SyntheticDataNID"
setwd(data.path)


m_list = c(15)
for (m in m_list) {
    g = pcalg::randomDAG(m, 0.2/(m/10)) # generated DAG
    moral_g = moralize(g) # moral graph
    edge.list = apply(get.edgelist(igraph.from.graphNEL(g)), c(1, 2), as.numeric)
    moral.edge.list = apply(get.edgelist(igraph.from.graphNEL(moral_g)), c(1, 2), as.numeric)
    file.name = paste("New_DAG_", m, ".txt", sep="")
    moral.file.name = paste("Moral_", file.name, sep="")
    write.table(edge.list, file.name, row.names=FALSE, col.names=FALSE)
    write.table(moral.edge.list, moral.file.name, row.names=FALSE, col.names=FALSE)
}

