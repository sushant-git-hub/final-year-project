import { polygonToCells, cellToLatLng, isValidCell } from 'h3-js';

/**
 * Generate H3-based candidate locations for a city
 *
 * @param {Array} boundaryGeoJson - GeoJSON Polygon coordinates
 * @param {number} resolution - H3 resolution (default 8)
 */
export function generateHexCandidates(boundaryGeoJson, resolution = 8) {
  console.log('DEBUG: generateHexCandidates called', { boundaryGeoJsonType: typeof boundaryGeoJson, isArray: Array.isArray(boundaryGeoJson), resolution });
  if (!boundaryGeoJson || !Array.isArray(boundaryGeoJson)) {
    throw new Error('Invalid city boundary');
  }

  // H3 expects [lat, lng]
  const h3Polygon = boundaryGeoJson.map(ring =>
    ring.map(([lng, lat]) => [lat, lng])
  );

  const hexIds = polygonToCells(
    h3Polygon,
    resolution
  );

  return hexIds
    .filter(isValidCell)
    .map(hexId => {
      const [lat, lng] = cellToLatLng(hexId);
      return {
        id: hexId,
        lat,
        lng
      };
    });
}