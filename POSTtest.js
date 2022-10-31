var axios = require('axios');
var data = '{\r\n  "m2m:sgn": {\r\n    "nev": {\r\n      "rep": {\r\n        "m2m:cin": {\r\n          "cnf": "application/json",\r\n          "con": {\r\n            "ae": "yt0",\r\n            "container": "location",\r\n            "wtime": 159649076010,\r\n            "lat": 31.859642,\r\n            "lng": 128.561136\r\n          }\r\n        }\r\n      }\r\n    }\r\n  }\r\n}';

var config = {
  method: 'post',
  url: 'http://192.168.1.166:1234/receiver',
  headers: { 
    'Accept': 'application/json', 
    'Content-Type': 'application/json;ty=4', 
    'X-M2M-RI': '1234'
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

