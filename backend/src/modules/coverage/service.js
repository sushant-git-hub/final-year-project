import { haversineDistanceKm } from '../../utils/geo.js';

export function computeCoverageScore(candidate, cityCenter, maxRadiusKm = 5) {
  const distance = haversineDistanceKm(
    candidate.lat,
    candidate.lng,
    cityCenter.lat,
    cityCenter.lng
  );

  // Inverse distance scoring
  const score = Math.max(0, 1 - distance / maxRadiusKm);

  return Number(score.toFixed(3));
}