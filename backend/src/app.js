import Fastify from 'fastify';
import cors from '@fastify/cors';
import analysisRoutes from './routes/analysis.routes.js';
import healthRoutes from './routes/health.routes.js';
import authRoutes from './routes/auth.route.js';
import predictRoutes from './routes/predict.js';

const app = Fastify({ logger: true });

// auth decorator (if any)
app.decorate('authenticate', async () => { });

// MIDDLEWARE
app.register(cors, {
    origin: true // allow all for dev
});

// REGISTER ROUTES
healthRoutes(app);
analysisRoutes(app);
predictRoutes(app);
// authRoutes(app);

export default app;