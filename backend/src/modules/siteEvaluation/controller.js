import { analyzeSite as analyzeSiteService } from './service.js';

export async function analyzeSite(req, reply) {
  const result = await analyzeSiteService(req.body);
  return reply.send(result);
}