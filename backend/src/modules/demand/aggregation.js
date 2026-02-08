import { latLngToCell } from 'h3-js';

/**
 * Aggregate orders into hex cells
 */
export function aggregateDemandOnHexes(orders, resolution = 8) {
  const hexMap = new Map();

  for (const order of orders) {
    const hexId = latLngToCell(order.lat, order.lng, resolution);

    const current = hexMap.get(hexId) || 0;
    hexMap.set(hexId, current + 1);
  }

  // Convert to array
  const results = Array.from(hexMap.entries()).map(
    ([hexId, orderCount]) => ({
      hexId,
      orderCount
    })
  );

  // Normalize demand score (0–1)
  const maxOrders = Math.max(...results.map(r => r.orderCount));

  return results.map(r => ({
    ...r,
    demandScore: maxOrders > 0 ? r.orderCount / maxOrders : 0
  }));
}