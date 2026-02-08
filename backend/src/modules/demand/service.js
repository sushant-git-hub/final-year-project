import { getOrdersByCity } from './repository.js';
import { aggregateDemandOnHexes } from './aggregation.js';

export async function getDemandByHex(cityName, resolution = 8) {
  const orders = await getOrdersByCity(cityName);

  if (!orders.length) {
    return [];
  }

  return aggregateDemandOnHexes(orders, resolution);
}