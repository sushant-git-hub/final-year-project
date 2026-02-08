export function computeSuitabilityScore({
  demandScore,
  coverageScore,
  costScore
}) {
  const weights = {
    demand: 0.45,
    coverage: 0.30,
    cost: 0.25
  };

  const rawScore =
    weights.demand * demandScore +
    weights.coverage * coverageScore +
    weights.cost * costScore;

  return Math.round(rawScore * 100);
}