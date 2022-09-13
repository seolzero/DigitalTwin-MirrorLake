const express = require("express");
const app = express();
app.listen(1111, () => {
    console.log("Server Start on port 1111");
 });
const redis = require('redis');

async function createRedisClient () {
  const client = redis.createClient();

  client.on('connect', () => console.log('Connected to REDIS!'));
  client.on('error', (err) => console.log('Error connecting to REDIS: ', err));

  await client.connect();
}

createRedisClient();


app.post("/redis/set", (req, res) => {
    client.set('ttt', 'hey', redis.print);  // redis.print : 수행결과 출력 혹은 오류 출력. redis.print는 없어도 상관없음. 없으면 결과 출력은 되지 않고 값만 저장

    res.end();
});

app.post("/redis/get", (req, res) => {
    client.get("ttt", function (err, value) {
        if (err) throw err;
        console.log(value);
     });
    res.send(value);
});