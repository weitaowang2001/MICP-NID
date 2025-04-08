# Run Wilcoxon rank sum test for the difference of our method and GES in Table 1.

setwd("/Users/tongxu/Downloads/projects/micodag")

results_ges <- read.csv('./experiment results/comparison to other benchmarks/1-30/ges_results_est_default_1_30.csv')
results_micodag <- read.csv('./experiment results/comparison to other benchmarks/1-30/MICP-NID-Perspective_est_12log(m)n_1_30.csv')


# Perform the Wilcoxon rank-sum test (Mann-Whitney U test)
graph.list <- c('1dsep', '2asia', '3bowling', '4insuranceSmall', '5rain', '6cloud', '7funnel', '8galaxy', '9insurance', '10factors', '11hfinder', '12hepar')

p_values <- data.frame(Graph = character(), P_Value = numeric(), Significant = character(), stringsAsFactors = FALSE)

# Loop through each graph and compute the p-value
for (graph in graph.list) {
  result_ges <- results_ges[results_ges$network == graph, ]
  result_micodag <- results_micodag[results_micodag$network == graph, ]
  
  test_result <- wilcox.test(result_ges$d_cpdag, result_micodag$d_cpdag, alternative = "greater")
  
  # Determine significance
  significance <- ifelse(test_result$p.value < 0.05, "Yes", "No")
  
  # Store results
  p_values <- rbind(p_values, data.frame(Graph = graph, P_Value = test_result$p.value, Significant = significance))
}

# Print the table
print(p_values)





# result_ges_1 <- results_ges[results_ges$network == '1dsep',]
# result_micodag_1 <- results_micodag[results_micodag$network == '1dsep',]
# result <- wilcox.test(result_ges_1$d_cpdag, result_micodag_1$d_cpdag, alternative = c("greater"))
# 
# # Print the result
# print(result)
# 
# # Check if the p-value is less than 0.05 for statistical significance
# if (result$p.value < 0.05) {
#   print("The difference between the two algorithms' performance is statistically significant.")
# } else {
#   print("The difference between the two algorithms' performance is not statistically significant.")
# }