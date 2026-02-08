export function estimateCostScore(candidate) {
  // Mock assumptions
  const baseCost = 1_000_000; // reference
  const demandPenalty = candidate.demand.demandScore * 0.4;
  const normalizedCost = Math.max(0, 1 - demandPenalty);

  return Number(normalizedCost.toFixed(3));
}