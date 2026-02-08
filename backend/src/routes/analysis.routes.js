import { analyzeSite } from '../modules/siteEvaluation/controller.js';

export default async function (app) {
  app.post(
    '/api/analyze-site',
    async (req, reply) => {
      return analyzeSite(req, reply);
    }
  );
}