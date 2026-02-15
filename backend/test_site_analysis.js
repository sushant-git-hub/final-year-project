
import axios from 'axios';

async function testAnalysis() {
    console.log("Starting analysis request...");
    const start = Date.now();
    try {
        const response = await axios.post('http://localhost:3005/api/analyze-site', {
            city: 'pune',
            type: 'retail',
            budget: 100000,
            radius: 5000
        });
        const duration = (Date.now() - start) / 1000;
        console.log(`Success! Status: ${response.status}`);
        console.log(`Duration: ${duration}s`);
        console.log(`Candidates: ${response.data.meta.candidateCount}`);
    } catch (error) {
        const duration = (Date.now() - start) / 1000;
        console.error(`Error! Status: ${error.response?.status}`);
        console.error(`Duration: ${duration}s`);
        console.error(error.message);
    }
}

testAnalysis();
