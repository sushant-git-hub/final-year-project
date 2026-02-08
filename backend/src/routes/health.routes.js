export default async function (app) {
  app.get('/health', async () => ({
    status: 'ok',
    service: 'mapmystore-backend',
    timestamp: new Date().toISOString()
  }));
}