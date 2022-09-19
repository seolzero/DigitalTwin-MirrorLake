const express = require('express');
const app = express();
const http = require("http");
const port = 1203;
const util = require('util');
// body parser
app.use(express.json());
app.use(
   express.urlencoded({
      extended: true,
   })
);

const server = http.createServer(app);
server.listen(port, () => {
   console.log(`Server Start on port ${port}`);
});
   
   
app.post('/simt' ,function(req, res){

   console.log(req.body);
   console.log(util.inspect(req.body, false, null, true));
   res.end();


});