import app from './app.js';

const PORT = process.env.PORT || 3005;

app.listen({ port: PORT, host: '0.0.0.0' }, (err, address) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
  console.log(`🚀 MapMyStore backend running at ${address}`);
});