import { MLAdapter } from './mlAdapter.js';

export class MockMLAdapter extends MLAdapter {
  async predict(features) {
    return features.map(f => ({
      hex_id: f.hex_id,
      demand_prediction: f.demand_score, // passthrough
      confidence: 0.5
    }));
  }
}