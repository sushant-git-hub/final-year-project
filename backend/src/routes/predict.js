import { runMLPredictions } from '../ml/mlService.js';

export default async function (fastify) {
    fastify.post('/api/predict', async (req, reply) => {
        const { latitude, longitude, category, features } = req.body;
        
        if (!latitude || !longitude) {
            return reply.code(400).send({ error: 'Latitude and Longitude are required' });
        }

        try {
            const prediction = await runMLPredictions({
                latitude,
                longitude,
                category,
                features
            });
            
            // Enhance response for frontend
            const response = {
                ...prediction,
                location: {
                    latitude,
                    longitude
                },
                display: {
                    success_percentage: `${(prediction.success_probability * 100).toFixed(1)}%`,
                    star_rating: Math.round(prediction.success_probability * 5),
                    recommendation: prediction.predicted_class === 1 ? 'RECOMMENDED' : 'NOT_RECOMMENDED',
                    risk_level: prediction.success_probability > 0.7 ? 'LOW' : 
                               (prediction.success_probability > 0.4 ? 'MEDIUM' : 'HIGH')
                }
            };

            return response;
        } catch (err) {
            req.log.error(err);
            reply.code(500).send({ error: 'Prediction failed', details: err.message });
        }
    });
}
