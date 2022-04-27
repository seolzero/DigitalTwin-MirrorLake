var axios = require('axios');
var data = JSON.stringify({
  "data": {
    "con": "29"
  }
});

var config = {
  method: 'post',
  url: 'http://192.168.10.9:1203/DigitalConnector/SensorGroup/power/te',
  headers: { 
    'Content-Type': 'application/json'
  },
  data : data
};

axios(config)
.then(function (response) {
  console.log(JSON.stringify(response.data));
})
.catch(function (error) {
  console.log(error);
});
