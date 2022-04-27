const express = require("express");
const app = express();
//const bodyParser = require('body-parser');
//app.use(bodyParser.json());
app.use(express.json());
app.use(express.urlencoded({extended : true}));

const port = 1203;
app.listen(port, () =>
   console.log(`Example app listening at http://localhost:${port}`)
);
const utf8 = require('utf8');


app.post("/sim", function (req, res) {

   console.log("> receivedData: ", req.body);

   console.log("> received URL: ", req.url, "\n", req.rawHeaders);

   res.status(200).send("post /end test ok");

});


app.post("/sim2", function (req, res) {

   console.log("> receivedData: ", req.body);

   console.log("> received URL: ", req.url, "\n", req.rawHeaders);

   res.status(200).send("post /end test ok");

});

app.post("/DigitalConnector/SensorGroup/:sensorName/te", function (req, res) {

   console.log("> receivedData: ", req.body);

   console.log("> received URL: ", req.url, "\n", req.rawHeaders);

   res.status(200).send("post /end test ok");

});

/*
app.post("/DigitalConnector/SensorGroup/:sensorName", function (req, res) {

   console.log("> receivedData: ", utf8.decode(req.body));

   console.log("> received URL: ", req.url, "\n", req.rawHeaders);

   res.status(200).send("post /end test ok");

});

*/
app.post("/DigitalConnector/SensorGroup/:sensorName", function (req, res) {
   let fullBody = "";

   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {

      console.log("> receivedData: ", fullBody);
      console.log("> received URL: ", req.url, "\n", req.rawHeaders);
      
      res.status(200).send("post /end test ok");
   });
});
