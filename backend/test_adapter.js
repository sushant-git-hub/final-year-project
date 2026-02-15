import { PythonServiceAdapter } from './src/ml/adapters/pythonServiceAdapter.js';

process.env.ML_SERVICE_URL = 'http://localhost:8001/predict';

const adapter = new PythonServiceAdapter();
console.log('Testing adapter...');

adapter.predict({
    latitude: 18.5204,
    longitude: 73.8567,
    category: 'retail',
    features: { competitor_count: 5 }
})
.then(res => console.log('Success:', JSON.stringify(res, null, 2)))
.catch(err => {
    console.error('Error:', err.message);
    if (err.response) {
        console.error('Response data:', err.response.data);
    }
});
