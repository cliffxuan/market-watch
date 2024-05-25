def weight_based_exposure(portfolio_weights, benchmark_weights):
  """
  This function calculates the weight-based exposure of a portfolio to a benchmark.

  Args:
      portfolio_weights: A dictionary where keys are asset names and values are their weights in the portfolio (0 to 1).
      benchmark_weights: A dictionary where keys are asset names (same as portfolio) and values are their weights in the benchmark (0 to 1).

  Returns:
      A float representing the overall exposure of the portfolio to the benchmark.
  """
  total_exposure = 0
  for asset in portfolio_weights:
    if asset in benchmark_weights:  # Check if asset exists in both
      portfolio_weight = portfolio_weights[asset]
      benchmark_weight = benchmark_weights[asset]
      exposure = portfolio_weight * benchmark_weight
      total_exposure += exposure
  return total_exposure

# Example Usage
portfolio_weights = {"Stock A": 0.4, "Stock B": 0.3, "Bond C": 0.3}
benchmark_weights = {"Stock A": 0.5, "Stock B": 0.2, "Stock C": 0.1, "Other": 0.2}  # Benchmark may have additional assets

exposure = weight_based_exposure(portfolio_weights, benchmark_weights)
print(f"Overall Exposure using Weighting Method: {exposure:.2f}")
