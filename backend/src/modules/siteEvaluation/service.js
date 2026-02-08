import { getCandidateLocations } from '../city/service.js';
import { demographicRepository } from '../demographics/repository.js';

export async function analyzeSite(criteria) {
    const { city, type, budget, radius } = criteria;

    const candidates = await getCandidateLocations(city);

    // 2. Score Candidates
    const scoredCandidates = candidates.map(hex => {
        // 2a. Fetch Demographics (Ward Info)
        const wardMetadata = demographicRepository.getNearestWard(hex.lat, hex.lng);
        const wardName = wardMetadata ? wardMetadata.name : 'Unknown';
        const population = wardMetadata ? wardMetadata.population : 12000; // Fallback

        // 2b. Feature Simulation (40+ features abstracted)
        // Base Random Factors with seeded-like behavior based on coords
        const randomFactor = (Math.sin(hex.lat * 1000) + Math.cos(hex.lng * 1000) + 2) / 4; // 0-1

        // Income Logic (Mock)
        const incomeMap = { 'low': 0.3, 'middle': 0.6, 'high': 0.9 };
        const areaIncomeLevel = randomFactor > 0.7 ? 'High' : (randomFactor > 0.4 ? 'Middle' : 'Low');

        // Proximity Logic (Mock)
        const nearMetro = randomFactor > 0.6;
        const nearMall = randomFactor > 0.8;

        // Demand Calculation
        // More population = more demand. Retail types need more footfall.
        let demandScore = (population / 100000) * 0.7 + randomFactor * 0.3;
        if (criteria.type === 'retail' || criteria.type === 'cafe') {
            demandScore += (nearMall || nearMetro ? 0.2 : 0);
        }
        demandScore = Math.min(0.98, Math.max(0.1, demandScore));

        // Cost Calculation
        // Higher demand = higher rent
        let numericBudget = Number(budget);
        if (isNaN(numericBudget) || numericBudget <= 0) {
            numericBudget = 100000; // Default budget if not provided
        }

        const baseRent = numericBudget * 0.1; // Base rent assumption
        const estimatedRent = baseRent + (demandScore * baseRent * 1.5) + (nearMetro ? 5000 : 0);

        // Calculate cost score relative to budget (1 if cheap, 0 if over budget)
        const costRatio = estimatedRent / numericBudget;
        const costScore = Math.max(0, 1 - Math.min(1, costRatio));

        // Coverage Score
        // (Placeholder for now)
        const coverageScore = 0.5 + (randomFactor * 0.4);

        // 2c. Success Score (ML Prediction Output)
        // Weighted combo of all factors
        const rawScore = (demandScore * 0.45) + (coverageScore * 0.30) + (costScore * 0.25);
        const successScore = Math.round(rawScore * 100);
        const successProbability = Math.min(99, Math.round(successScore * 0.9 + (randomFactor * 10)));

        // 2d. Revenue Estimation
        // Simple formula: Footfall * Conversion * AvgTicket
        const estimatedFootfall = Math.round(population * 0.05 * demandScore); // Daily
        const avgTicket = criteria.type === 'cafe' ? 300 : 800;
        const monthlyRevenue = estimatedFootfall * 30 * 0.1 * avgTicket; // 10% conversion

        // 2e. Key Factors
        const keyFactors = [];
        if (nearMetro) keyFactors.push("Near Metro Station");
        if (nearMall) keyFactors.push("High Commercial Activity");
        if (population > 50000) keyFactors.push("Dense Population");
        if (costScore > 0.7) keyFactors.push("Affordable Rent");
        if (demandScore > 0.8) keyFactors.push("High Footfall Area");
        if (keyFactors.length === 0) keyFactors.push("Steady Growth Potential");

        return {
            id: hex.id,
            lat: hex.lat,
            lng: hex.lng,

            // Core ML Outputs
            suitabilityScore: successScore,
            successScore: successScore,
            successProbability: successProbability, // %
            expectedRevenue: Math.round(monthlyRevenue),

            // Feature Breakdown
            demand: {
                demandScore: parseFloat(demandScore.toFixed(2)),
                orderCount: Math.round(estimatedFootfall * 30)
            },
            coverageScore: parseFloat(coverageScore.toFixed(2)),
            costScore: parseFloat(costScore.toFixed(2)),

            // Rich Metadata
            ward: wardName,
            population: population,
            keyFactors: keyFactors.slice(0, 3), // Top 3
            metrics: {
                rent: Math.round(estimatedRent),
                footfall: estimatedFootfall,
                competition: randomFactor > 0.8 ? "High" : "Moderate"
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
