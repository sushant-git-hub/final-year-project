
import axios from 'axios';

async function testMLBatchDirectly() {
    console.log("Testing ML Service Batch Endpoint directly...");
    const start = Date.now();
    
    // Create dummy locations (10 points)
    const locations = Array(10).fill(0).map((_, i) => ({
        latitude: 18.5204 + (i * 0.001),
        longitude: 73.8567 + (i * 0.001),
        category: 'retail',
        features: {}
    }));

    try {
        const response = await axios.post('http://127.0.0.1:8001/predict-batch', {
            locations: locations
        });
        const duration = (Date.now() - start) / 1000;
        console.log(`Success! Status: ${response.status}`);
        console.log(`Duration: ${duration}s`);
        console.log(`Predictions: ${response.data.predictions.length}`);
        console.log('Sample Prediction:', JSON.stringify(response.data.predictions[0], null, 2));
    } catch (error) {
        const duration = (Date.now() - start) / 1000;
        console.error(`Error! Status: ${error.response?.status}`);
        console.error(`Duration: ${duration}s`);
        console.error(error.message);
        if (error.response) {
            console.error('Data:', JSON.stringify(error.response.data, null, 2));
        }
    }
}

testMLBatchDirectly();
