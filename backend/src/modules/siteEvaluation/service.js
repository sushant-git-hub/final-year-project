import { getCandidateLocations } from '../city/service.js';
import { demographicRepository } from '../demographics/repository.js';
import { runBatchMLPredictions } from '../../ml/mlService.js';

export async function analyzeSite(criteria) {
    const { city, type, budget, radius } = criteria;

    const candidates = await getCandidateLocations(city);

    // 2. Score Candidates
    console.time('PreparePayloads');
    // 2. Score Candidates with ML Model
    // Prepare locations for batch prediction
    const locationPayloads = candidates.map(hex => {
        // Fetch Demographics (Ward Info)
        const wardMetadata = demographicRepository.getNearestWard(hex.lat, hex.lng);
        const population = wardMetadata ? wardMetadata.population : 12000;
        
        // Prepare features (matching what the ML model expects)
        // In a real scenario, these would come from a rich feature store.
        // For now, we'll use some logical defaults and the demographic info we have.
        
        return {
            latitude: hex.lat,
            longitude: hex.lng,
            category: type, // 'retail' or 'food'
            features: {
                // Core demographic features
                total_population: population,
                
                // We'll trust the ML model lookup for most geo-features
                // But we can override if we have specific data
            }
        };
    });
    console.timeEnd('PreparePayloads');

    // 2a. Call ML Service
    let predictions = [];
    try {
        console.time('MLBatchPrediction');
        console.log(`Calling ML service with ${locationPayloads.length} locations...`);
        predictions = await runBatchMLPredictions(locationPayloads);
        console.timeEnd('MLBatchPrediction');
    } catch (error) {
        console.error("ML Batch Prediction failed, falling back to basic scoring:", error.message);
        // Fallback or empty array - for now let's return error to debug
        throw error;
    }

    // 2b. Merge Predictions with Candidates
    const scoredCandidates = candidates.map((hex, index) => {
        const prediction = predictions[index] || {};
        const wardMetadata = demographicRepository.getNearestWard(hex.lat, hex.lng);
        const wardName = wardMetadata ? wardMetadata.name : 'Unknown';
        const population = wardMetadata ? wardMetadata.population : 12000;

        // Extract ML outputs
        // Handle cases where prediction might have failed for a specific row
        const successProbability = prediction.success_probability || 0;
        const predictedClass = prediction.predicted_class || 0;
        const confidenceLevel = prediction.confidence_level || 'LOW';
        
        // Scale success probability to a 0-100 score
        const successScore = Math.round(successProbability * 100);

        // Revenue Estimation (can be refined with ML output if available, otherwise heuristic)
        // Simple formula: Footfall * Conversion * AvgTicket
        // We can use successProbability as a proxy for 'demand'
        const estimatedFootfall = Math.round(population * 0.05 * successProbability); // Daily
        const avgTicket = type === 'cafe' ? 300 : 800;
        const monthlyRevenue = estimatedFootfall * 30 * 0.1 * avgTicket; 

        // Cost Calculation (Heuristic based on location/success)
        let numericBudget = Number(budget);
        if (isNaN(numericBudget) || numericBudget <= 0) {
            numericBudget = 100000; 
        }
        // Higher success prob usually implies higher rent
        const baseRent = numericBudget * 0.1;
        const estimatedRent = baseRent + (successProbability * baseRent * 1.5);
        const costRatio = estimatedRent / numericBudget;
        const costScore = Math.max(0, 1 - Math.min(1, costRatio));

        // Key Factors from ML (if available) or Heuristics
        const keyFactors = [];
        if (prediction.confidence_level === 'VERY_HIGH') keyFactors.push("High Model Confidence");
        if (successProbability > 0.8) keyFactors.push("High Success Probability");
        if (population > 50000) keyFactors.push("Dense Population");
        if (costScore > 0.7) keyFactors.push("Affordable Rent");
        if (keyFactors.length === 0) keyFactors.push("Moderate Potential");

        // Format for frontend
        return {
            id: hex.id,
            lat: hex.lat,
            lng: hex.lng,

            // Core ML Outputs
            suitabilityScore: successScore,
            successScore: successScore,
            successProbability: Math.round(successProbability * 100), 
            expectedRevenue: Math.round(monthlyRevenue),

            // Feature Breakdown
            demand: {
                demandScore: parseFloat(successProbability.toFixed(2)),
                orderCount: Math.round(estimatedFootfall * 30)
            },
            coverageScore: 0.8, // Placeholder
            costScore: parseFloat(costScore.toFixed(2)),

            // Rich Metadata
            ward: wardName,
            population: population,
            keyFactors: keyFactors.slice(0, 3),
            metrics: {
                rent: Math.round(estimatedRent),
                footfall: estimatedFootfall,
                competition: "Moderate" // We could get this from ML features if exposed
            }
        };
    });

    // 3. Sort by Suitability
    scoredCandidates.sort((a, b) => b.suitabilityScore - a.suitabilityScore);

    // 4. Generate Heatmap Data (from demand scores)
    const heatmap = scoredCandidates.map(c => ({
        lat: c.lat,
        lng: c.lng,
        intensity: c.demand.demandScore
    }));

    return {
        meta: {
            city,
            candidateCount: scoredCandidates.length,
            timestamp: new Date().toISOString()
        },
        rankings: scoredCandidates.slice(0, 50), // Top 50
        heatmap
    };
}
