// Mock historical orders for now
export async function getOrdersByCity(cityName) {
  return [
    { lat: 28.6139, lng: 77.2090 },
    { lat: 28.6145, lng: 77.2100 },
    { lat: 28.5641, lng: 77.1872 },
    { lat: 28.6038, lng: 77.2383 },
    { lat: 28.6039, lng: 77.2384 },
    { lat: 28.6037, lng: 77.2385 }
  ];
}