# Using this 
library(pcalg)
library(igraph)
library(gRbase)

data.path = "/Users/tongxu/Downloads/projects/micodag/Data/SyntheticDataNID_30"
setwd(data.path)


m_list = c(35)
for (m in m_list) {
    g = pcalg::randomDAG(m, 2/m) # generated DAG
    moral_g = moralize(graph_from_graphnel(g)) # moral graph
    edge.list = apply(as_edgelist(graph_from_graphnel(g)), c(1, 2), as.numeric)
    moral.edge.list = apply(as_edgelist(moral_g), c(1, 2), as.numeric)
    file.name = paste("DAG_", m, ".txt", sep="")
    moral.file.name = paste("Moral_", file.name, sep="")
    write.table(edge.list, file.name, row.names=FALSE, col.names=FALSE)
    write.table(moral.edge.list, moral.file.name, row.names=FALSE, col.names=FALSE)
}
edge.list

