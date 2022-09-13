
const express = require('express');
const app = express();

var requestIp = require('request-ip');

const port = 7576;
app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`))



app.post("*", (req, res)=>{
   console.log(req.originalUrl);

   var fullBody = '';
   req.on('data', function(chunk) {
      fullBody += chunk; 
   });

   req.on('end', function() {
      res.status(200).send('post /end test ok');
      var receivedData = JSON.parse(fullBody);
      console.log(receivedData['m2m:cin'].con);

   });

})
