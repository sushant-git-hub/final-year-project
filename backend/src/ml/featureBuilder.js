export function buildLocationFeatures(candidate, cityMeta) {
  return {
    hex_id: candidate.id,

    // demand features
    demand_score: candidate.demand.demandScore,
    order_count: candidate.demand.orderCount,

    // spatial features
    lat: candidate.lat,
    lng: candidate.lng,

    // coverage proxy
    distance_to_center_km: cityMeta.distanceFn(candidate),

    // cost proxy
    cost_score: candidate.costScore
  };
}