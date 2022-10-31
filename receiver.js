const express = require("express");
const app = express();
const http = require("http");
const port = 1234;

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

app.post("/receiver", function (req, res) {
   console.log(req.body);

   res.send({ temperature: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
});

