export function attachReasons(ranked) {
  return ranked.map(loc => ({
    ...loc,
    reasons: [
      loc.components.demand.demandScore > 0.7 && 'High demand',
      loc.components.cost.withinBudget && 'Cost within budget',
      loc.components.coverage.coverageScore > 0.6 && 'Strong delivery coverage'
    ].filter(Boolean)
  }));
}