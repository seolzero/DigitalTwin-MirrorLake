const express = require("express");
const app = express();

const port = 1203;
app.listen(port, () =>
   console.log(`Example app listening at ${port}`)
);

app.post("*", (req, res) => {
   var fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", function () {
      res.status(200).send("post /end test ok");
      var receivedData = JSON.parse(fullBody);
      //var rep = receivedData['m2m:sgn'].nev.rep;
      console.log("> receivedData: ", receivedData);
      //console.log("> rep: ", rep);
   });
});
