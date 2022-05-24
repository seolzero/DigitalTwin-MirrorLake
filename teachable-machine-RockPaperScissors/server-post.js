const express = require("express");
const bp = require("body-parser");
const app = express();
const port = 1212;

app.use(bp.json());
app.use(bp.urlencoded({ extended: true }));

app.get("/", (req, res) => {
   res.send("Hello World!");
});

app.post("/test", (req, res) => {
   console.log("req: \n", req.body);
   //res.send("Hello World!");
   res.status(200).send({ ssul: req.body });
   //res.json({ ssul: req.body });
});

app.listen(port, () => {
   console.log(`Example app listening at http://localhost:${port}`);
});
