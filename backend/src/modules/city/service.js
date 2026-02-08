import { cityRepository } from './repository.js';
import { generateHexCandidates } from './candidateGenerator.js';

export async function getCandidateLocations(cityName) {
  console.log('DEBUG: getCandidateLocations called for', cityName);
  const city = await cityRepository.getCityByName(cityName);
  console.log('DEBUG: City found:', city ? city.name : 'null');

  if (!city) {
    throw new Error(`City not supported: ${cityName}`);
  }

  if (!city.boundary) {
    throw new Error(`Boundary missing for city: ${cityName}`);
  }

  const candidates = generateHexCandidates(
    city.boundary,
    city.hexResolution || 8
  );

  if (!candidates) {
    console.error('CRITICAL: generateHexCandidates returned undefined for', city.name);
    throw new Error('Failed to generate candidates');
  }

  console.log('DEBUG: candidates generated:', candidates.length);
  return candidates;
}