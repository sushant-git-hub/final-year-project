import { cellToLatLng } from 'h3-js';

/**
 * Convert hex demand into heatmap points
 */
export function buildHeatmapFromHexDemand(hexDemand) {
  if (!Array.isArray(hexDemand)) return [];

  return hexDemand.map(d => {
    const [lat, lng] = cellToLatLng(d.hexId);

    return {
      lat,
      lng,
      intensity: Number(d.demandScore.toFixed(3))
    };
  });
}