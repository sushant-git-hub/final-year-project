export function rank(scoredLocations) {
  return scoredLocations
    .sort((a, b) => b.score - a.score)
    .map(loc => ({
      ...loc,
      label:
        loc.score >= 75 ? 'High'
        : loc.score >= 50 ? 'Medium'
        : 'Risky'
    }));
}