const config = require("./configs.json");
const express = require("express");
const app = express();
const axios = require("axios");
const unirest = require("unirest");
const util = require("util");
const where = require("lodash.where");
const { DateTime } = require("luxon");
const mqtt = require("mqtt");
app.listen(1005, () => {
   console.log("Server Start on port 1209");
});
const { tryJSONparse } = require("./lib");
let Options = config.ksqlOptions;
const redis = require("redis");
const Rclient = redis.createClient({
   port: config.redis.port,
   host: config.redis.ip,
});

Rclient.on("error", function (error) {
   console.error(error);
});

const client = mqtt.connect(`mqtt://${config.mqtt.ip}:${config.mqtt.port}`);
// var options = { retain:true, qos:1 }; //client.publish(topic, dataToString, options);

client.on("connect", function () {
   console.log("connected!", client.connected);
});
client.on("error", (error) => {
   console.log("Can't connect" + error);
   process.exit(1);
});

let SESSIONID = "";
const gwOptions = {
   hostname: config.gwOption.hostname,
   port: config.gwOption.port,
};
app.post("/flink/:session", (req, res) => {
   SESSIONID = req.params.session;
   console.log("sessionID: ", SESSIONID);
   res.end();
});

function create_flink_DO_table(DOobject) {
   console.log("create DO Table");
   let DOName = DOobject.name;
   gwOptions.path = `/v1/sessions/${SESSIONID}/statements`;
   gwOptions.method = POST;

   // Create DO Table
   let createStreamSQL = {
      statement: ``,
   };

   // Get Sensor List from DO Object
   let sensorList = DOobject.sensor;
   console.log(sensorList);

   if (sensorList.length == 1) {
      createStreamSQL.statement = `CREATE TABLE ${DOName}(tmpA BIGINT, name STRING, data STRING, rowtime TIMESTAMP(3), PRIMARY KEY (tmpA) NOT ENFORCED) WITH ('connector' = 'upsert-kafka', 'topic' = 'DO_${DOName}','properties.bootstrap.servers' = '${config.kafkaHost}', 'key.format' = 'json', 'value.format' = 'json')`;
   } else {
      createStreamSQL.statement = `CREATE TABLE ${DOName} (tmpA BIGINT, name STRING, rowtime TIMESTAMP(3), `;

      for (i = 0; i < sensorList.length; i++) {
         createStreamSQL.statement += `${sensorList[i]} STRING, `;
      }

      createStreamSQL.statement += `PRIMARY KEY (tmpA) NOT ENFORCED) WITH('connector' = 'upsert-kafka', 'topic' = 'DO_${DOName}','properties.bootstrap.servers' = '${config.kafkaHost}', 'key.format' = 'json', 'value.format' = 'json')`;
   }

   console.log("createStreamSQL: ", createStreamSQL);

   let insertTableSQL = {
      statement: `INSERT INTO ${DOName} select `,
   };

   if (sensorList.length == 1) {
      insertTableSQL.statement += `${sensorList[0]}.tmp, '${DOName}', ${sensorList[0]}.sensor_value, ${sensorList[0]}.sensor_rowtime FROM ${sensorList[0]}`;
   } else {
      insertTableSQL.statement += `${sensorList[0]}.tmp, '${DOName}', ${sensorList[0]}.sensor_rowtime, `;

      for (i = 0; i < sensorList.length; i++) {
         insertTableSQL.statement += `${sensorList[i]}.sensor_value `;
         if (i != sensorList.length - 1) {
            insertTableSQL.statement += `, `;
         } else if (i == sensorList.length - 1) {
            insertTableSQL.statement += `from  ${sensorList[0]} `;
         }
      }

      for (i = 0; i < sensorList.length - 1; i++) {
         insertTableSQL.statement += `left join ${
            sensorList[i + 1]
         } for system_time as of ${sensorList[0]}.sensor_rowtime on ${
            sensorList[i + 1]
         }.tmp=${sensorList[0]}.tmp `;
      }
   }

   console.log("insertTableSQL: ", insertTableSQL);

   //Send Request to sql-gateway Server
   var request = http.request(gwOptions, function (response) {
      let fullBody = "";

      response.on("data", function (chunk) {
         fullBody += chunk;
      });

      response.on("end", function () {
         console.log(fullBody);
         console.log("Insert Sensor Table to DO Table");

         var insertRequest = http.request(gwOptions, function (insertResponse) {
            let fullBody = "";

            insertResponse.on("data", function (chunk) {
               fullBody += chunk;
            });

            insertResponse.on("end", function () {
               console.log(fullBody);
            });

            insertResponse.on("error", function (error) {
               console.error(error);
            });
         });
         insertRequest.write(JSON.stringify(insertTableSQL));
         insertRequest.end();
         // res.status(201).json(req.body);
      });

      response.on("error", function (error) {
         console.error(error);
      });
   });
   request.write(JSON.stringify(createStreamSQL));
   request.end();
}
/*
 * DO Creation
 */
app.post("/DigitalTwin/DO", async function (req, res) {
   let fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DOWholeData = tryJSONparse(fullBody);
         if (
            DOWholeData?.name &&
            DOWholeData?.sensor &&
            DOWholeData.sensor.length > 0
         ) {
            const flag = await checkNameExist(DOWholeData.name, "DO").then(
               function (flag) {
                  return flag;
               }
            );
            if (flag) {
               res.status(500).send("DO is already exist");
            } else {
               const DOName = DOWholeData.name;
               Rclient.rpush("DO", DOName);
               const DOobject = CheckKeyExistAndAddCount(DOWholeData);
               Rclient.set(DOName, JSON.stringify(DOobject));
               create_flink_DO_table(DOobject);
               res.status(200).send(DOobject);
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});

function getNameList(key) {
   return new Promise((resolve) => {
      Rclient.lrange(key, 0, -1, function (err, keys) {
         if (err) throw err;
         resolve(keys);
      });
   });
}
async function checkNameExist(Name, key) {
   let NameList = await getNameList(key).then((List) => {
      return List;
   });
   let flag = false;
   return new Promise((resolve, reject) => {
      for (i in NameList) {
         if (NameList[i] == Name) {
            flag = true;
         }
      }
      resolve(flag);
   });
}
function CheckKeyExistAndAddCount(DOWholeData) {
   if (Object.keys(DOWholeData).some((v) => v == "sensor")) {
      DOWholeData.sensorCount = DOWholeData.sensor.length;
   }
   if (Object.keys(DOWholeData).some((v) => v == "control")) {
      DOWholeData.controlCount = DOWholeData.control.length;
   }
   DOWholeData.creationTime = new Date().getTime();
   return DOWholeData;
}

/*
 * DO Retrieve
 */
app.get("/DigitalTwin/DO/:DOName", async (req, res) => {
   if (req.params.DOName) {
      let DOName = req.params.DOName;
      const flag = await checkNameExist(DOName, "DO").then(function (flag) {
         return flag;
      });
      if (flag) {
         Rclient.get(DOName, function (err, value) {
            if (err) throw err;
            res.status(200).send(JSON.parse(value));
         });
      } else {
         res.status(200).send("Unregistered DO");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

/*
 * DO Entire Retrieve
 */
app.get("/DigitalTwin/DO/list", async function (req, res) {
   await getNameList("DO").then((List) => {
      res.status(200).send(List);
   });
});

/*
 * DO DELETE
 */
app.delete("/DigitalTwin/DO/:DOName", async (req, res) => {
   if (req.params.DOName) {
      let DOName = req.params.DOName;
      const flag = await checkNameExist(DOName, "DO").then(function (flag) {
         return flag;
      });
      if (flag) {
         Rclient.DEL(DOName);
         Rclient.LREM("DO", -1, DOName);
         res.send({ success: 1 });
      } else {
         res.status(200).send("Unregistered DO");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

app.delete("/DigitalTwin/DO/all", (req, res) => {
   Rclient.DEL("DO");
   DONameListDelete();
   res.send({ success: 1 });
});

async function DONameListDelete() {
   let NameList = await getNameList("DO").then((List) => {
      return List;
   });
   let flag = true;
   return new Promise((resolve, reject) => {
      for (i in NameList) {
         Rclient.DEL(i);
      }
      resolve(flag);
   });
}

/*
 * DO UPDATE
 */
app.put("/DigitalTwin/DO", async (req, res) => {
   var fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DOWholeData = tryJSONparse(fullBody);
         if (
            DOWholeData?.name &&
            DOWholeData?.sensor &&
            DOWholeData.sensor.length > 0
         ) {
            const flag = await checkNameExist(DOWholeData.name, "DO").then(
               function (flag) {
                  return flag;
               }
            );
            if (!flag) {
               res.status(500).send("Unregistered DO");
            } else {
               const DOName = DOWholeData.name;
               const DOobject = CheckKeyExistAndAddCount(DOWholeData);
               console.log("DO: ", DOobject);
               Rclient.set(DOName, JSON.stringify(DOobject));
               postDOobjectToKSQL(DOobject); //post DOobject
               res.status(200).send("update DO");
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});

//========================================================================>> SERVICE
/*
 * service Creation
 */
app.post("/DigitalTwin/serviceGroup", function (req, res) {
   let fullBody = "",
      DataObject = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DataObject = tryJSONparse(fullBody);
         if (DataObject?.name && DataObject?.url) {
            const flag = await checkNameExist(DataObject.name, "service").then(
               function (flag) {
                  return flag;
               }
            );
            if (flag) {
               res.status(500).send("is already exist");
            } else {
               const service = DataObject.name;
               Rclient.rpush("service", service);
               const sensorFields = Object.keys(DataObject);
               for (var i = 0; i < sensorFields.length; i++) {
                  const field = sensorFields[i];
                  Rclient.hset(
                     `service_${DataObject.name}`,
                     sensorFields[i],
                     JSON.stringify(DataObject[field])
                  );
               }
               res.status(200).send("successfully registered");
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});

/*
 * service Update
 */
app.put("/DigitalTwin/serviceGroup", function (req, res) {
   let fullBody = "",
      DataObject = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DataObject = tryJSONparse(fullBody);
         if (DataObject?.name && DataObject?.url) {
            const flag = await checkNameExist(DataObject.name, "service").then(
               function (flag) {
                  return flag;
               }
            );
            if (!flag) {
               res.status(500).send("Unregistered service");
            } else {
               const service = DataObject.name;
               const sensorFields = Object.keys(DataObject);
               for (var i = 0; i < sensorFields.length; i++) {
                  const field = sensorFields[i];
                  Rclient.hset(
                     `service_${DataObject.name}`,
                     sensorFields[i],
                     JSON.stringify(DataObject[field])
                  );
               }
               res.status(200).send("successfully update");
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});
//get hash table
function getKeys(Name) {
   return new Promise((resolve) => {
      Rclient.hkeys(Name, function (err, keys) {
         resolve(keys);
      });
   });
}
//get hash table fliel value
function getValue(Name, key) {
   return new Promise((resolve, reject) => {
      Rclient.hget(Name, key, function (err, value) {
         if (err) reject(err);
         resolve(JSON.parse(value));
      });
   });
}
/*
 * service Retrieve
 */
app.get("/DigitalTwin/serviceGroup/:serviceName", async (req, res) => {
   if (req.params.serviceName) {
      let serviceName = req.params.serviceName;
      const flag = await checkNameExist(serviceName, "service").then(function (
         flag
      ) {
         return flag;
      });
      if (flag) {
         const keys = await getKeys(`service_${req.params.serviceName}`).then(
            function (keys) {
               return keys;
            }
         );

         let resObject = {};
         for (let key of keys) {
            const value = await getValue(
               `service_${req.params.serviceName}`,
               key
            );
            resObject[key] = value;
         }
         res.status(200).send(resObject);
      } else {
         res.status(200).send("Unregistered service");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

/*
 * service delete
 */
app.delete("/DigitalTwin/serviceGroup/:serviceName", async (req, res) => {
   if (req.params.serviceName) {
      let serviceName = req.params.serviceName;
      const flag = await checkNameExist(serviceName, "service").then(function (
         flag
      ) {
         return flag;
      });
      if (flag) {
         Rclient.DEL(`service_${req.params.serviceName}`);
         Rclient.LREM("service", -1, req.params.serviceName);
         deleteSink(req.params.serviceName);
         res.send({ success: 1 });
      } else {
         res.status(200).send("Unregistered service");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

/*
 * service Trigger
 */
app.post("/DigitalTwin/serviceGroup/trigger/:serviceName", function (req, res) {
   let fullBody = "",
      serviceName = "",
      resObject = {};
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (req.params.serviceName) {
         serviceName = req.params.serviceName;
         const flag = await checkNameExist(serviceName, "service").then(
            function (flag) {
               return flag;
            }
         );
         if (flag) {
            const keys = await getKeys(`service_${serviceName}`).then(function (
               keys
            ) {
               return keys;
            });
            for (let key of keys) {
               const value = await getValue(`service_${serviceName}`, key);
               resObject[key] = value;
            }
            CreateServiceSinkConnector(resObject);
            res.status(200).send(resObject);
         } else {
            res.status(412).send("Unregistered service");
         }
      } else {
         res.status(500).send("please check serviceName parameter");
      }
   });
});

//=======================================================================================>> simulation

/*
 * simulation Creation
 */
app.post("/DigitalTwin/simulationGroup", function (req, res) {
   let fullBody = "",
      DataObject = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DataObject = tryJSONparse(fullBody);
         if (DataObject?.name && DataObject?.arg && DataObject?.url) {
            const flag = await checkNameExist(
               DataObject.name,
               "simulation"
            ).then(function (flag) {
               return flag;
            });
            if (flag) {
               res.status(500).send("is already exist");
            } else {
               const simulation = DataObject.name;
               Rclient.rpush("simulation", simulation);
               const sensorFields = Object.keys(DataObject);
               for (var i = 0; i < sensorFields.length; i++) {
                  const field = sensorFields[i];
                  Rclient.hset(
                     `simulation_${DataObject.name}`,
                     field,
                     JSON.stringify(DataObject[field])
                  );
               }
               res.status(200).send(DataObject);
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});

/*
 * simulation Update
 */
app.put("/DigitalTwin/simulationGroup", function (req, res) {
   let fullBody = "",
      DataObject = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", async function () {
      if (tryJSONparse(fullBody)) {
         DataObject = tryJSONparse(fullBody);
         if (DataObject?.name && DataObject?.arg && DataObject?.url) {
            const flag = await checkNameExist(
               DataObject.name,
               "simulation"
            ).then(function (flag) {
               return flag;
            });
            if (!flag) {
               res.status(500).send("Unregistered simulation");
            } else {
               const simulation = DataObject.name;
               const sensorFields = Object.keys(DataObject);
               for (var i = 0; i < sensorFields.length; i++) {
                  const field = sensorFields[i];
                  Rclient.hset(
                     `simulation_${DataObject.name}`,
                     sensorFields[i],
                     JSON.stringify(DataObject[field])
                  );
               }
               res.status(200).send("successfully update");
            }
         } else {
            res.status(500).send("please check mandatory field");
         }
      } else {
         res.status(500).send("is not a json structure");
      }
   });
});

/*
 * simulation Retrieve
 */
app.get("/DigitalTwin/simulationGroup/:simulationName", async (req, res) => {
   if (req.params.simulationName) {
      let simulationName = req.params.simulationName;
      const flag = await checkNameExist(simulationName, "simulation").then(
         function (flag) {
            return flag;
         }
      );
      if (flag) {
         const keys = await getKeys(
            `simulation_${req.params.simulationName}`
         ).then(function (keys) {
            return keys;
         });

         let resObject = {};
         for (let key of keys) {
            const value = await getValue(
               `simulation_${req.params.simulationName}`,
               key
            );
            resObject[key] = value;
         }
         res.status(200).send(resObject);
      } else {
         res.status(200).send("Unregistered simulation");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

/*
 * simulation delete
 */
app.delete("/DigitalTwin/simulationGroup/:simulationName", async (req, res) => {
   if (req.params.simulationName) {
      let simulationName = req.params.simulationName;
      const flag = await checkNameExist(simulationName, "simulation").then(
         function (flag) {
            return flag;
         }
      );
      if (flag) {
         Rclient.DEL(`simulation_${req.params.simulationName}`);
         Rclient.LREM("simulation", -1, req.params.simulationName);
         deleteSink(req.params.simulationName);
         res.send({ success: 1 });
      } else {
         res.status(200).send("Unregistered simulation");
      }
   } else {
      res.status(404).send("Bad Request");
      console.log("input value error");
   }
});

function deleteSink(connectorName) {
   console.log("Delete Sink Connector");
   var config = {
      method: "delete",
      url: `http://${kafkaConnectServer.hostname}:${kafkaConnectServer.port}/connectors/${connectorName}`,
      headers: {},
   };

   axios(config)
      .then(function (response) {
         console.log(JSON.stringify(response.data));
      })
      .catch(function (error) {
         console.log(error);
      });
}

app.delete("/DigitalTwin/simulationGroup/all", async (req, res) => {
   Rclient.DEL("simulation");
   let NameList = await getNameList("simulation").then((List) => {
      return List;
   });
   new Promise((resolve, reject) => {
      for (i in NameList) {
         Rclient.DEL(i);
      }
      resolve(NameList);
   });

   res.send(NameList);
});
/*
 * simulation Trigger
 * RT: RealTime
 */
app.post(
   "/DigitalTwin/simulationGroup/RTtrigger/:simName",
   function (req, res) {
      let fullBody = "",
         simName = "",
         resObject = {};
      req.on("data", function (chunk) {
         fullBody += chunk;
      });

      req.on("end", async function () {
         if (req.params.simName) {
            simName = req.params.simName;
         } else {
            res.status(500).send("please check simName parameter");
         }
         if (req.params.simName) {
            let simName = req.params.simName;
            const flag = await checkNameExist(simName, "simulation").then(
               function (flag) {
                  return flag;
               }
            );
            if (flag) {
               const keys = await getKeys(`simulation_${simName}`).then(
                  function (keys) {
                     return keys;
                  }
               );
               for (let key of keys) {
                  const value = await getValue(`simulation_${simName}`, key);
                  resObject[key] = value;
               }
               console.log(
                  `createRTSink: `,
                  util.inspect(resObject, false, null, true)
               );
               CreateSimulationSinkConnector(resObject);
               res.status(200).send(resObject);
            } else {
               res.status(200).send("Unregistered simulation");
            }
         } else {
            res.status(500).send("please check simName parameter");
         }
      });
   }
);

/*
 * simulation Trigger
 * ST: Static Time
 */
app.post(
   "/DigitalTwin/simulationGroup/STtrigger/:simName",
   function (req, res) {
      let fullBody = "",
         DataObject = "",
         simName = "",
         resObject = {};
      req.on("data", function (chunk) {
         fullBody += chunk;
      });

      req.on("end", async function () {
         if (req.params.simName) {
            simName = req.params.simName;
         } else {
            res.status(500).send("please check simName parameter");
         }
         if (tryJSONparse(fullBody)) {
            DataObject = tryJSONparse(fullBody);
            if (DataObject?.data) {
               const flag = await checkNameExist(simName, "simulation").then(
                  function (flag) {
                     return flag;
                  }
               );
               if (flag) {
                  const keys = await getKeys(`simulation_${simName}`).then(
                     function (keys) {
                        return keys;
                     }
                  );
                  for (let key of keys) {
                     const value = await getValue(`simulation_${simName}`, key);
                     resObject[key] = value;
                  }
                  //resObject.url로 DataObject전송
                  console.log(
                     `createSTSink:  url => ${
                        resObject.url
                     } , data => ${util.inspect(DataObject.data)}`
                  );
                  //CreateSinkConnector(resObject);
                  res.status(200).send(
                     `createSTSink:  url => ${
                        resObject.url
                     } , data => ${util.inspect(DataObject.data)}`
                  );
               } else {
                  res.status(200).send("Unregistered simulation");
               }
            } else {
               res.status(500).send("please check mandatory field");
            }
         } else {
            res.status(500).send("is not a json structure");
         }
      });
   }
);
//=======================================================================>> control

/*
 * control Creation
 */
app.post("/DigitalTwin/:DOName/control", function (req, res) {
   var fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", function () {
      let DOName = req.params.DOName;
      var controlNameObject;
      controlNameObject = JSON.parse(fullBody);

      if (DONameList.includes(DOName)) {
         console.log("body: ", controlNameObject, "DOName: ", DOName);
         DOWholeDataList.forEach((element, index) => {
            if (element.name == DOName) {
               if (element.control) {
                  var filtered = where(element.control, {
                     name: controlNameObject.name,
                  });
                  if (filtered[0]) {
                     res.status(400).send("control is already exist");
                     console.log("same name exist: ", filtered[0]);
                     console.log("element: ", element);
                  } else {
                     res.status(200).send("Received control Data");
                     element.control.push(controlNameObject);
                     element.controlCount++;
                     console.log("push: ", element);
                  }
               } else {
                  res.status(200).send("Received control Data");
                  element.control = [controlNameObject];
                  element.controlCount = 1;
                  console.log(
                     "control push: ",
                     util.inspect(element, false, null, true)
                  );
               }
            }
         });
      } else {
         res.status(404).send("DO does not exist");
      }
   });
});

/*
 * control data Creation
 */
app.post("/DigitalTwin/:DOName/controlData", function (req, res) {
   var fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", function () {
      let DOName = req.params.DOName;
      var controlDataObject;
      controlDataObject = JSON.parse(fullBody);

      if (DONameList.includes(DOName)) {
         console.log("body: ", controlDataObject, "DOName: ", DOName);
         DOWholeDataList.forEach((element, index) => {
            if (element.name == DOName) {
               if (element.control) {
                  var filtered = where(element.control, {
                     name: controlDataObject.name,
                  });
                  if (filtered[0]) {
                     res.status(200).send("Received control Data");
                     // console.log("control is exist: ", filtered[0]);
                     // console.log("element: ", util.inspect(element, false, null, true));
                     if (filtered[0].data) {
                        filtered[0].data.push(controlDataObject.data); //check please!!
                        var fifoControlDataPushArray = new FifoArray(
                           5,
                           filtered[0].data
                        );
                        filtered[0].data = fifoControlDataPushArray;
                        console.log(
                           "create data arr & push data: ",
                           util.inspect(element, false, null, true)
                        );
                        var controlDataElementToString =
                           JSON.stringify(element);
                        client.publish(
                           "dp_do_data",
                           controlDataElementToString
                        ); //send string text!
                        Rclient.set(key_DO, JSON.stringify(value));
                     } else {
                        filtered[0].data = [controlDataObject.data];
                        console.log(
                           "push data: ",
                           util.inspect(element, false, null, true)
                        );
                        var controlDataElementToString =
                           JSON.stringify(element);
                        client.publish(
                           "dp_do_data",
                           controlDataElementToString
                        ); //send string text!
                        Rclient.set(key_DO, JSON.stringify(value));
                     }
                  } else {
                     res.status(404).send("The control name does not exist");
                  }
               } else {
                  // control create tag does not exist
                  res.status(404).send("A DO with no control created");
               }
            }
         });
      } else {
         res.status(404).send("DO does not exist");
      }
   });
});

/*
 * control result update
 */
app.put("/DigitalTwin/:DOName/:controlName", function (req, res) {
   var fullBody = "";
   req.on("data", function (chunk) {
      fullBody += chunk;
   });

   req.on("end", function () {
      let DOName = req.params.DOName;
      let controlName = req.params.controlName;
      var controlUpdateDataObject;
      controlUpdateDataObject = JSON.parse(fullBody);
      if (DONameList.includes(DOName)) {
         res.status(200).send("Received control Data");
         DOWholeDataList.forEach((element, index) => {
            if (element.name == DOName) {
               if (element.control) {
                  var filtered = where(element.control, { name: controlName });
                  if (filtered[0]) {
                     if (filtered[0].data) {
                        filtered[0].data.forEach((data, index) => {
                           const controlData = data.toString();
                           let controlDataStringArr = controlData.split(", ");
                           const updateControlData =
                              controlUpdateDataObject.data.toString();
                           let updateControlDataStringArr =
                              updateControlData.split(", ");

                           if (
                              controlDataStringArr[0] ==
                                 updateControlDataStringArr[0] &&
                              controlDataStringArr[1] ==
                                 updateControlDataStringArr[1]
                           ) {
                              if (controlUpdateDataObject.controlDone) {
                                 let dataString =
                                    controlUpdateDataObject.data +
                                    ", " +
                                    controlUpdateDataObject.controlReceived +
                                    ", " +
                                    controlUpdateDataObject.controlDone;
                                 filtered[0].data.splice(index, 1, dataString);
                                 let controlDataSetToString =
                                    JSON.stringify(element);
                                 client.publish(
                                    "dp_do_data",
                                    controlDataSetToString
                                 );
                                 Rclient.set(key_DO, JSON.stringify(value));
                                 console.log(
                                    "Update controlReceived, controlDone: ",
                                    util.inspect(element, false, null, true)
                                 );
                              } else {
                                 let dataString =
                                    controlUpdateDataObject.data +
                                    ", " +
                                    controlUpdateDataObject.controlReceived;
                                 filtered[0].data.splice(index, 1, dataString);
                                 let controlDataSetToString =
                                    JSON.stringify(element);
                                 client.publish(
                                    "dp_do_data",
                                    controlDataSetToString
                                 );
                                 Rclient.set(key_DO, JSON.stringify(value));
                                 console.log(
                                    "Update controlReceived: ",
                                    util.inspect(element, false, null, true)
                                 );
                              }
                           }
                        });
                     }
                  }
               }
            }
         });
      } else {
         res.status(404).send("DO does not exist");
      }
   });
});

/*
 * control delete
 */
app.delete("/DigitalTwin/:DOName/control/:controlName", (req, res) => {
   let DOName = req.params.DOName;
   let controlName = req.params.controlName;
   console.log(DOName, controlName);
   if (DONameList.includes(DOName)) {
      DOWholeDataList.forEach((element, index) => {
         if (element.name == DOName) {
            if (element.control) {
               var filtered = where(element.control, { name: controlName });
               if (filtered[0]) {
                  var controlIndex = element.control.findIndex(
                     (i) => i.name == controlName
                  );
                  element.control.splice(controlIndex, 1);
                  element.controlCount--;
                  console.log(
                     "element: ",
                     util.inspect(element, false, null, true)
                  );

                  res.status(200).send(`control ${controlName} delete`);
               } else {
                  res.status(200).send("control does not exist");
                  console.log(
                     "element: ",
                     util.inspect(element, false, null, true)
                  );
               }
            } else {
               res.status(404).send("control object does not exist");
               console.log("control does not exist");
               console.log(
                  "element: ",
                  util.inspect(element, false, null, true)
               );
            }
         }
      });
   } else {
      res.status(404).send("DO does not exist");
   }
});

//=============================================================================> KAFKA SINK
const url = require("url");
const http = require("http");
const POST = "post";
const GET = "get";
const DELETE = "delete";
const PUT = "put";

let kafkaConnectServer = `http://${config.kafkaconnectHost}/connectors`; //create connector address
kafkaConnectServer = url.parse(kafkaConnectServer, true);
var options = {
   hostname: kafkaConnectServer.hostname,
   port: kafkaConnectServer.port,
   path: kafkaConnectServer.pathname,
   headers: {
      "Content-Type": "application/json",
   },
   maxRedirects: 20,
   method: POST,
};

/**
 * CreateSinkConnector => service, simulation
 * MQTT, HTTP
 */

async function CreateServiceSinkConnector(resObject) {
   let sinkConnectorBody;
   const { name, url, DO_arg, SIM_arg } = { ...resObject };
   console.log("resObject", resObject);
   let splitURLsink = url.split(":");
   switch (splitURLsink[0]) {
      case "http":
         sinkConnectorBody = await ServiceHttpSinkConnector(resObject);
         console.log("http sink");
         break;
      case "mqtt":
         sinkConnectorBody = await ServiceMQTTSinkConnector(
            resObject,
            splitURLsink
         );
         console.log("mqtt sink");
         break;
      default:
         console.log(`out of ${splitURLsink[0]}`);
   }

   console.log("sinkConnectorBody\n", sinkConnectorBody);
   /**
    * Send Request to Kafka Connect Server
    */
   var request = http.request(options, function (response) {
      var chunks = [];

      response.on("data", function (chunk) {
         chunks.push(chunk);
      });

      response.on("end", function (chunk) {
         var body = Buffer.concat(chunks);
         console.log(body.toString());
      });

      response.on("error", function (error) {
         console.error(error);
      });
   });
   request.write(JSON.stringify(sinkConnectorBody));
   request.end();
}

function ServiceHttpSinkConnector(resObject) {
   const DOs = Object.keys(resObject.DO_arg); //[ 'DO1', 'DO2' ]
   const SIMs = Object.keys(resObject.SIM_arg);
   let topics = "";
   if (SIMs.length > 0) {
      let SIM_SIMs = SIMs.map((s) => "SIM_" + s);
      for (i in SIM_SIMs) {
         topics += SIM_SIMs[i];
         if (i != SIM_SIMs.length - 1) {
            topics += ",";
         }
      }
   } else {
      let DO_s = DOs.map((s) => "DO_" + s);
      for (i in DO_s) {
         topics += DO_s[i];
         if (i != DO_s.length - 1) {
            topics += ",";
         }
      }
   }

   let sinkConnectorBody = {
      name: resObject.name,
      config: {
         "connector.class":
            "uk.co.threefi.connect.http.service.HttpSinkConnector",
         "tasks.max": "1",
         headers: "Content-Type:application/json|Accept:application/json",
         "key.converter": "org.apache.kafka.connect.storage.StringConverter",
         "value.converter": "org.apache.kafka.connect.storage.StringConverter",
         "http.api.url": resObject.url,
         "request.method": "POST",
         topics: topics,
      },
   };

   return sinkConnectorBody;
}

function ServiceMQTTSinkConnector(resObject, splitURLsink) {
   const DOs = Object.keys(resObject.DO_arg); //[ 'DO1', 'DO2' ]
   const SIMs = Object.keys(resObject.SIM_arg);
   let DO_DOs = DOs.map((d) => "DO_" + d);
   let SIM_SIMs = SIMs.map((s) => "SIM_" + s);
   let DO_SIM_arr = DO_DOs.concat(SIM_SIMs);
   //console.log(DO_SIM_arr);

   let topics = "";
   for (i in DO_SIM_arr) {
      topics += DO_SIM_arr[i];
      if (i != DO_SIM_arr.length - 1) {
         topics += ",";
      }
   }
   //console.log(topics);

   let SQL = "";
   for (i in DO_DOs) {
      SQL += `INSERT INTO /mqtt/data SELECT * FROM ${DO_DOs[i]};`;
   }

   for (i in SIM_SIMs) {
      SQL += `INSERT INTO /mqtt/simulation SELECT * FROM ${SIM_SIMs[i]};`;
   }

   let sinkConnectorBody = {
      name: resObject.name,
      config: {
         "connector.class":
            "com.datamountaineer.streamreactor.connect.mqtt.sink.MqttSinkConnector",
         "tasks.max": "1",
         topics: topics,
         "connect.mqtt.hosts": `tcp:${splitURLsink[1]}:${splitURLsink[2]}`,
         "connect.mqtt.clean": "true",
         "connect.mqtt.timeout": "1000",
         "connect.mqtt.keep.alive": "1000",
         "connect.mqtt.service.quality": "1",
         "key.converter": "org.apache.kafka.connect.json.JsonConverter",
         "key.converter.schemas.enable": "false",
         "value.converter": "org.apache.kafka.connect.json.JsonConverter",
         "value.converter.schemas.enable": "false",
         "connect.mqtt.kcql": SQL,
      },
   };

   //console.log("sinkConnectorBody\n", sinkConnectorBody);
   return sinkConnectorBody;
}

async function CreateSimulationSinkConnector(resObject) {
   let sinkConnectorBody;
   const { name, url, arg } = { ...resObject };
   console.log("resObject", resObject);
   let splitURLsink = url.split(":");
   switch (splitURLsink[0]) {
      case "http":
         sinkConnectorBody = await SimulationHttpSinkConnector(resObject);
         console.log("http sink");
         break;
      case "mqtt":
         sinkConnectorBody = await SimulationMQTTSinkConnector(
            resObject,
            splitURLsink
         );
         console.log("mqtt sink");
         break;
      default:
         console.log(`out of ${splitURLsink[0]}`);
   }

   console.log("sinkConnectorBody\n", sinkConnectorBody);
   /**
    * Send Request to Kafka Connect Server
    */
   var request = http.request(options, function (response) {
      var chunks = [];

      response.on("data", function (chunk) {
         chunks.push(chunk);
      });

      response.on("end", function (chunk) {
         var body = Buffer.concat(chunks);
         console.log(body.toString());
      });

      response.on("error", function (error) {
         console.error(error);
      });
   });
   request.write(JSON.stringify(sinkConnectorBody));
   request.end();
}

function SimulationHttpSinkConnector(resObject) {
   const DOs = Object.keys(resObject.arg); //[ 'DO1', 'DO2' ]
   console.log("DOs: ", DOs);
   let DO_DOs = DOs.map((d) => "DO_" + d);
   console.log("DO_DOs: ", DO_DOs);
   let topics = "";
   for (i in DO_DOs) {
      topics += DO_DOs[i];
      if (i != DO_DOs.length - 1) {
         topics += ",";
      }
   }
   console.log("topics", topics);

   let sinkConnectorBody = {
      name: resObject.name,
      config: {
         "connector.class": "uk.co.threefi.connect.http.HttpSinkConnector",
         "tasks.max": "1",
         headers: "Content-Type:application/json|Accept:application/json",
         "key.converter": "org.apache.kafka.connect.storage.StringConverter",
         "value.converter": "org.apache.kafka.connect.storage.StringConverter",
         "http.api.url": resObject.url,
         "request.method": "POST",
         topics: topics,
         "response.topic": `SIM_${resObject.name}`,
         "kafka.api.url": `${config.kafkaHost}`,
         "batch.max.size": 512,
      },
   };

   return sinkConnectorBody;
}

function SimulationMQTTSinkConnector(resObject, splitURLsink) {
   const DOs = Object.keys(resObject.arg); //[ 'DO1', 'DO2' ]
   let DO_DOs = DOs.map((d) => "DO_" + d);

   let topics = "";
   for (i in DO_DOs) {
      topics += DO_DOs[i];
      if (i != DO_DOs.length - 1) {
         topics += ",";
      }
   }
   console.log(topics);
   let SQL = "";
   for (i in DO_DOs) {
      SQL += `INSERT INTO /mqtt/data SELECT * FROM ${DO_DOs[i]};`;
   }
   let sinkConnectorBody = {
      name: resObject.name,
      config: {
         "connector.class":
            "com.datamountaineer.streamreactor.connect.mqtt.sink.MqttSinkConnector",
         "tasks.max": "1",
         topics: topics,
         "connect.mqtt.hosts": `tcp:${splitURLsink[1]}:${splitURLsink[2]}`,
         "connect.mqtt.clean": "true",
         "connect.mqtt.timeout": "1000",
         "connect.mqtt.keep.alive": "1000",
         "connect.mqtt.service.quality": "1",
         "key.converter": "org.apache.kafka.connect.json.JsonConverter",
         "key.converter.schemas.enable": "false",
         "value.converter": "org.apache.kafka.connect.json.JsonConverter",
         "value.converter.schemas.enable": "false",
         "connect.mqtt.kcql": SQL,
      },
   };
   return sinkConnectorBody;
}

//====================================The 404 Route (ALWAYS Keep this as the last route)
app.get("*", function (req, res) {
   res.send("Bad Request (Wrong Url)", 404);
});

app.post("*", function (req, res) {
   res.send("Bad Request (Wrong Url)", 404);
});

app.delete("*", function (req, res) {
   res.send("Bad Request (Wrong Url)", 404);
});

app.put("*", function (req, res) {
   res.send("Bad Request (Wrong Url)", 404);
});
