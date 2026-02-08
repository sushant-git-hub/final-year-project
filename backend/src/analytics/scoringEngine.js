export function computeScores({ demandSignals, costSignals, coverageSignals }) {
  return demandSignals.map(demand => {
    const cost = costSignals.find(c => c.locationId === demand.locationId);
    const coverage = coverageSignals.find(c => c.locationId === demand.locationId);

    const score =
      0.45 * demand.demandScore +
      0.30 * normalizeCost(cost) +
      0.25 * coverage.coverageScore;

    return {
      locationId: demand.locationId,
      score: Math.round(score * 100),
      components: { demand, cost, coverage }
    };
  });
}