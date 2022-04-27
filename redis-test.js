const redis = require('redis');

async function createRedisClient () {
  const client = redis.createClient();

  client.on('connect', () => console.log('Connected to REDIS!'));
  client.on('error', (err) => console.log('Error connecting to REDIS: ', err));

  await client.connect();

  return client;
}

module.exports = createRedisClient();

