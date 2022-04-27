var request = require('request');
var options = {
  'method': 'POST',
  'url': 'http://172.20.175.144:8083/connectors',
  'headers': {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "name": "test-res",
    "config": {
      "connector.class": "uk.co.threefi.connect.http.HttpSinkConnector",
      "tasks.max": "1",
      "http.api.url": "http://192.168.10.9:1203/sim2",
      "topics": "test-res",
      "request.method": "POST",
      "headers": "Content-Type:application/json|Accept:application/json",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "value.converter": "org.apache.kafka.connect.storage.StringConverter",
      "response.topic": "test-response"
    }
  })

};
request(options, function (error, response) {
  if (error) throw new Error(error);
  console.log(response.body);
});
