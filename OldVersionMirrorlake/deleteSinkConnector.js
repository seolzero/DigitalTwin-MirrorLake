var request = require('request');
var options = {
  'method': 'DELETE',
  'url': 'http://172.20.175.144:8083/connectors/test-res',
  'headers': {
  }
};
request(options, function (error, response) {
  if (error) throw new Error(error);
  console.log(response.body);
});
