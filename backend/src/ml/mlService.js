import { MockMLAdapter } from './adapters/mockAdapter.js';
import { PythonServiceAdapter } from './adapters/pythonServiceAdapter.js';

const adapter =
  process.env.ML_MODE === 'python'
    ? new PythonServiceAdapter()
    : new MockMLAdapter();

export async function runMLPredictions(features) {
  return adapter.predict(features);
}

export async function runBatchMLPredictions(locations) {
    return adapter.predictBatch(locations);
}